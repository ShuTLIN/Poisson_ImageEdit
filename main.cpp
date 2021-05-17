#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>
#include <iostream>
#include <math.h>
#include <map>


#include <Dense>
#include <Sparse>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::VectorXd Vec;
using namespace std;

int xyn2oneD(int x, int y, int n, int iw) {
    return x*3+n+(y*iw*3);
}

// gamma correction constant.
constexpr float GAMMA = 2.2f;

class vec3 {
private:
    float x, y, z;
public:
    vec3(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
    vec3(float v) { this->x = v; this->y = v; this->z = v; }
    vec3() { this->x = this->y = this->z = 0; }
    vec3& operator+=(const vec3& b) { (*this) = (*this) + b; return (*this); }
    friend vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
    friend vec3 operator+(const vec3& a, const vec3& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
    friend vec3 operator*(const float s, const vec3& a) { return vec3(s * a.x, s * a.y, s * a.z); }
    friend vec3 operator*(const vec3& a, const float s) { return s * a; }
    const float& operator[] (int index)const { return ((float*)(this))[index]; }
    float& operator[] (int index) { return ((float*)(this))[index]; }
};

float clamp(float x) {
    if (x > 1.0f) {
        return 1.0f;
    }
    else if (x < 0.0f) {
        return 0.0f;
    }
    else {
        return x;
    }
}

class loadImg {
    public:
        int width, hight, n;
        unsigned char* channel1;
        unsigned char* channel2;
        unsigned char* channel3;
        unsigned char* idata;
        loadImg(string path) {
            idata = stbi_load(path.c_str(), &width, &hight, &n, 0);
            cout << "w: " << width << ", h: " << hight << ", n: " << n << endl;
        }

        void splitChannel() {
            auto* red = (unsigned char*)malloc(width * hight );
            auto* green = (unsigned char*)malloc(width * hight);
            auto* blue = (unsigned char*)malloc(width * hight);
            for (int j = 0; j < hight; j++) {
                for (int i = 0; i < width; i++) {
                    red[i   + j * width] = idata[i * 3 + 0 + j * width * 3];
                    green[i   + j * width] = idata[i * 3 + 1 + j * width * 3];
                    blue[i   + j * width ] = idata[i * 3 + 2 + j * width * 3];
                }
            }
            channel1 = red;
            channel2 = green;
            channel3 = blue;
        }
};

int targetFlatten(unsigned int x, unsigned int y, int targetImage_width) {
    return  targetImage_width * y*3 + x*3;
}

unsigned int maskFlatten(unsigned int x, unsigned int y, int maskImage_width) {
    return  maskImage_width * y*3 + x*3;
}

// check if pixel is part in mask. pixels with a red RGB value of 1.0 are part of the mask. Note that we also have a small margin, though.
bool isMaskPixel(unsigned int x, unsigned int y,loadImg maskImage) {
    return maskImage.idata[maskFlatten(x, y, maskImage.width)] > 255.0*0.9;
}

// compute image gradient.
float vpq(
    float fpstar, float fqstar,
    float gp, float gq) {
    float fdiff = fpstar - fqstar;
    float gdiff = gp - gq;

    // equation (11) in the paper.
    return gdiff;

}

int main(int argc, char **argv)
{
    string inputPath = "pool-target.jpg";
    string sourcePath = "bear.jpg";
    string maskPath = "bear-mask.jpg";
    
    //int iw, ih, n;
    
    loadImg target(inputPath);
    loadImg source(sourcePath);
    loadImg source_mask(maskPath);
    target.splitChannel();
    

    unsigned int mx=0;
    unsigned int my=0;
    
    std::map<unsigned int, unsigned int> varMap;
    {
        int i = 0;
        for (unsigned int y = 0; y < source_mask.hight; ++y) {
            for (unsigned int x = 0; x < source_mask.width; ++x) {
                if (isMaskPixel(x, y, source_mask)) {
                    varMap[maskFlatten(x, y, source_mask.width)] = i;
                    ++i;
                }
            }
        }
    }
    const unsigned int numUnknowns = (unsigned int)varMap.size();

    std::vector<Triplet> mt; // M triplets. sparse matrix entries of M matrix.
    {
        unsigned int irow = 0;
        for (unsigned int y = my; y < my + source_mask.hight; ++y) {
            for (unsigned int x = mx; x < mx + source_mask.width; ++x) {
                if (isMaskPixel(x - mx, y - my, source_mask)) {
                   
                    mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - my, source_mask.width)], 4)); // |N_p| = 4.

                    
                    if (isMaskPixel(x - mx, y - my - 1, source_mask)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - 1 - my, source_mask.width)], -1));
                    }
                    if (isMaskPixel(x - mx + 1, y - my, source_mask)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx + 1, y - my, source_mask.width)], -1));
                    }
                    if (isMaskPixel(x - mx, y - my + 1, source_mask)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - my + 1, source_mask.width)], -1));
                    }
                    if (isMaskPixel(x - mx - 1, y - my, source_mask)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx - 1, y - my, source_mask.width)], -1));
                    }

                    ++irow; // jump to the next row in the matrix.
                }
            }
        }
    }

    Eigen::SimplicialCholesky<SpMat> solver;
    {
        SpMat mat(numUnknowns, numUnknowns);
        mat.setFromTriplets(mt.begin(), mt.end());
        solver.compute(mat);
    }

    Vec solutionChannels[3];
    Vec b(numUnknowns);

    for (unsigned int ic = 0; ic < 3; ++ic)
    {
        /*
        For each of the three color channels RGB, there will be a different b vector.
        So to perform poisson blending on the entire image, we must solve for x three times in a row, one time for each channel.

        */

        unsigned int irow = 0;

        for (unsigned int y = my; y < my + source.hight; ++y) {
            for (unsigned int x = mx; x < mx + source.width; ++x) {

                if (isMaskPixel(x - mx, y - my,source_mask)) {
                    // we only ended up using v in the end.
                    int v = source.idata[maskFlatten(x - mx, y - my,source_mask.width)];
                    int u = target.idata[targetFlatten(x, y,target.width)];

                    /*
                    The right-hand side of (7) determines the value of b.
                    below, we sum up all the values of v_pq(the gradient) for all neighbours.
                    */
                    float grad =
                        vpq(
                            u, target.idata[targetFlatten(x, y - 1, target.width)], // unused
                            v, source.idata[maskFlatten(x - mx, y - 1 - my, source_mask.width)]) // used
                        +
                        vpq(
                            u, target.idata[targetFlatten(x - 1, y, target.width)], // unused
                            v, source.idata[maskFlatten(x - 1 - mx, y - my, source_mask.width)]) // used
                        +
                        vpq(
                            u, target.idata[targetFlatten(x, y + 1, target.width)], // unused
                            v, source.idata[maskFlatten(x - mx, y + 1 - my, source_mask.width)] // used
                        )
                        +
                        vpq(
                            u, target.idata[targetFlatten(x + 1, y, target.width)], // unused
                            v, source.idata[maskFlatten(x + 1 - mx, y - my, source_mask.width)]); // used

                    b[irow] = grad;

                    /*
                    Finally, due to the boundary condition, some values of f_q end up on the right-hand-side, because they are not unknown.

                    The ones outside the mask end up here.
                    */
                    if (!isMaskPixel(x - mx, y - my - 1, source_mask)) {
                        b[irow] += target.idata[targetFlatten(x, y - 1, target.width)];
                    }
                    if (!isMaskPixel(x - mx + 1, y - my, source_mask)) {
                        b[irow] += target.idata[targetFlatten(x + 1, y, target.width)];
                    }
                    if (!isMaskPixel(x - mx, y - my + 1, source_mask)) {
                        b[irow] += target.idata[targetFlatten(x, y + 1, target.width)];
                    }
                    if (!isMaskPixel(x - mx - 1, y - my, source_mask)) {
                        b[irow] += target.idata[targetFlatten(x - 1, y, target.width)];
                    }

                    ++irow;
                }
            }
        }

        // solve for channel number ic.
        solutionChannels[0] = solver.solve(b);
    }


    auto *odata = (unsigned char *) malloc(target.width * target.hight * target.n);
    
    for(int j=0; j< target.hight; j++) {
        for(int i=0; i< target.width; i++) {
               /* odata[i*3+0+j* target.width *3] = target.idata[i*3+0+j*  target.width *3];
                odata[i*3+1+j* target.width *3] = target.idata[i*3+1+j* target.width *3];
                odata[i*3+2+j*target.width *3] = target.idata[i*3+2+j*  target.width *3];*/
                odata[i * 3 + 0 + j * target.width * 3] = target.channel1[i  + j * target.width ];
                odata[i * 3 + 1 + j * target.width * 3] = target.channel1[i  + j * target.width ];
                odata[i * 3 + 2 + j * target.width * 3] = target.channel1[i  + j * target.width ];
            }
    }
    

    for(int j = 0; j < source_mask.hight; j++) {
        for(int i = 0; i < source_mask.width; i++) {
           /* if (((int)source_mask.idata[i * 3 + 0 + j * source_mask.width * 3] >= 255*0.9)) {
                odata[i * 3 + 0 + j * target.width * 3] = source.idata[i * 3 + 0 + j * source.width * 3];
                odata[i * 3 + 1 + j * target.width * 3] = source.idata[i * 3 + 1 + j * source.width * 3];
                odata[i * 3 + 2 + j * target.width * 3] = source.idata[i * 3 + 2 + j * source.width * 3];
            }*/
            if (isMaskPixel(i, j,source_mask)) {
                unsigned int k = varMap[maskFlatten(i, j,source_mask.width)];
                float col = solutionChannels[0][k];
                //std::cout << col << std::endl;
                if (col < 0) {
                    col = 0.0;
                }
                //col = clamp(col);
                
                odata[i * 3 + 0 + j * target.width * 3] = (unsigned char)col;
                odata[i * 3 + 1 + j * target.width * 3] = (unsigned char)col;
                odata[i * 3 + 2 + j * target.width * 3] = (unsigned char)col;
            }
        }
    }

    string outputPath = "out.bmp";
    cout << (int)source_mask.idata[65 * 3 + 0 + 50 * 107 * 3] << endl;
    // write
    stbi_write_png(outputPath.c_str(), target.width, target.hight, target.n, odata, 0);

    stbi_image_free(target.idata);
    stbi_image_free(odata);

   /* Eigen::Matrix3f A;
    Eigen::Vector3f b;
    A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    b << 3, 3, 4;
    cout << "Here is the matrix A:\n" << A << endl;
    cout << "Here is the vector b:\n" << b << endl;
    Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
    cout << "The solution is:\n" << x << endl;

    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;*/
    cin.get();
   return 0;
}