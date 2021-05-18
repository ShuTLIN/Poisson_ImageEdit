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
#define Poisson 0

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::VectorXd Vec;
using namespace std;

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

class loadImg {
public:
    int width, hight, n;
    unsigned char* idata;
    loadImg(string path) {
        idata = stbi_load(path.c_str(), &width, &hight, &n, 0);
        cout << "w: " << width << ", h: " << hight << ", n: " << n << endl;
    }
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

unsigned int targetFlatten(unsigned int x, unsigned int y, unsigned int targetImage_width) {
    return  targetImage_width * y*3 + x*3;
}

unsigned int maskFlatten(unsigned int x, unsigned int y, unsigned int maskImage_width) {
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

    //gradient
    return gdiff;

}

void Poisson_restore(loadImg target,unsigned char* odata) {
    std::map<unsigned int, unsigned int> varMap;
    int i = 0;
    for (unsigned int y = 0; y < target.hight; ++y) {
        for (unsigned int x = 0; x < target.width; ++x) {
            if (isMaskPixel(x, y, target)) {
                varMap[maskFlatten(x, y, target.width)] = i;
                ++i;
            }
        }
    }

    unsigned int numUnknowns = (unsigned int)varMap.size();

    std::vector<Triplet> mt; // M triplets. sparse matrix entries of M matrix.
    {
        unsigned int irow = 0;
        for (unsigned int y = 0; y < target.hight; ++y) {
            for (unsigned int x = 0; x < target.width; ++x) {
                if (isMaskPixel(x , y , target)) {

                    mt.push_back(Triplet(irow, varMap[maskFlatten(x , y, target.width)], 4)); // |N_p| = 4.


                    if (isMaskPixel(x , y  - 1, target)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x , y - 1, target.width)], -1));
                    }
                    if (isMaskPixel(x  + 1, y, target)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x  + 1, y, target.width)], -1));
                    }
                    if (isMaskPixel(x , y  + 1, target)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x , y  + 1, target.width)], -1));
                    }
                    if (isMaskPixel(x - 1, y , target)) {
                        mt.push_back(Triplet(irow, varMap[maskFlatten(x - 1, y , target.width)], -1));
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

    Vec solutionChannels;
    Vec b(numUnknowns);

        /*
        For each of the three color channels RGB, there will be a different b vector.
        So to perform poisson blending on the entire image, we must solve for x three times in a row, one time for each channel.
        */
        unsigned int irow = 0;

        for (unsigned int y =0; y <target.hight; ++y) {
            for (unsigned int x =0; x < target.width; ++x) {

                if (isMaskPixel(x , y , target)) {
                    // we only ended up using v in the end.
                   
                    float u = (float)target.idata[targetFlatten(x, y, target.width)] / 255.0;

                    /*
                    sum up all the values of v_pq(the gradient) for all neighbours.
                    */
                    float grad = 0;
                        

                    b[irow] = grad;

                    /*
                    due to the boundary condition, some values of f_q end up on the right-hand-side, because they are not unknown.
                    The ones outside the mask end up here.
                    */
                    if (!isMaskPixel(x , y  - 1, target)) {
                        b[irow] += (float)target.idata[targetFlatten(x, y - 1, target.width)] / 255.0;
                    }
                    if (!isMaskPixel(x  + 1, y , target)) {
                        b[irow] += (float)target.idata[targetFlatten(x + 1, y, target.width)] / 255.0;
                    }
                    if (!isMaskPixel(x , y  + 1, target)) {
                        b[irow] += (float)target.idata[targetFlatten(x, y + 1, target.width)] / 255.0;
                    }
                    if (!isMaskPixel(x  - 1, y , target)) {
                        b[irow] += (float)target.idata[targetFlatten(x - 1, y, target.width)] / 255.0;
                    }

                    ++irow;

                }
            }
        }

        // solve for channel number ic.
        solutionChannels = solver.solve(b);
        for (int j = 0; j < target.hight; j++) {
            for (int i = 0; i < target.width; i++) {
                if (isMaskPixel(i, j, target)) {
                    unsigned int k = varMap[maskFlatten(i, j, target.width)];
                    float col = solutionChannels[k];

                    odata[i * 3 + 0 + j * target.width * 3] = (unsigned char)(col * 255.0);
                    odata[i * 3 + 1 + j * target.width * 3] = (unsigned char)(col * 255.0);
                    odata[i * 3 + 2 + j * target.width * 3] = (unsigned char)(col * 255.0);
                }
            }
        }

    }



int main(int argc, char **argv)
{
    string inputPath = "street.jpg";
    string restorePath = "einsteinSample.bmp";
    string sourcePath = "pupu.jpg";
    string maskPath = "pupu_mask.jpg";

    
    loadImg target(inputPath);
    loadImg source(sourcePath);
    loadImg source_mask(maskPath);
    loadImg restore_target(restorePath);
   
  
    //offset
    unsigned int mx = 90;
    unsigned int my = 520;

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

                if (isMaskPixel(x - mx, y - my, source_mask)) {
                    // we only ended up using v in the end.
                    float v = (float)source.idata[maskFlatten(x - mx, y - my, source_mask.width) + ic] / 255.0;
                    float u = (float)target.idata[targetFlatten(x, y, target.width) + ic] / 255.0;

                    /*
                    sum up all the values of v_pq(the gradient) for all neighbours.
                    */
                    float grad =
                        vpq(
                            u, target.idata[targetFlatten(x, y - 1, target.width) + ic], // unused
                            v, (float)source.idata[maskFlatten(x - mx, y - 1 - my, source_mask.width) + ic] / 255.0) // used
                        +
                        vpq(
                            u, target.idata[targetFlatten(x - 1, y, target.width) + ic], // unused
                            v, (float)source.idata[maskFlatten(x - 1 - mx, y - my, source_mask.width) + ic] / 255.0) // used
                        +
                        vpq(
                            u, target.idata[targetFlatten(x, y + 1, target.width) + ic], // unused
                            v, (float)source.idata[maskFlatten(x - mx, y + 1 - my, source_mask.width) + ic] / 255.0) // used
                        +
                        vpq(
                            u, target.idata[targetFlatten(x + 1, y, target.width) + ic], // unused
                            v, (float)source.idata[maskFlatten(x + 1 - mx, y - my, source_mask.width) + ic] / 255.0); // used

                    b[irow] = grad;

                    /*
                    due to the boundary condition, some values of f_q end up on the right-hand-side, because they are not unknown.
                    The ones outside the mask end up here.
                    */
                    if (!isMaskPixel(x - mx, y - my - 1, source_mask)) {
                        b[irow] += (float)target.idata[targetFlatten(x, y - 1, target.width) + ic] / 255.0;
                    }
                    if (!isMaskPixel(x - mx + 1, y - my, source_mask)) {
                        b[irow] += (float)target.idata[targetFlatten(x + 1, y, target.width) + ic] / 255.0;
                    }
                    if (!isMaskPixel(x - mx, y - my + 1, source_mask)) {
                        b[irow] += (float)target.idata[targetFlatten(x, y + 1, target.width) + ic] / 255.0;
                    }
                    if (!isMaskPixel(x - mx - 1, y - my, source_mask)) {
                        b[irow] += (float)target.idata[targetFlatten(x - 1, y, target.width) + ic] / 255.0;
                    }

                    ++irow;

                }
            }
        }

        // solve for channel number ic.
        solutionChannels[ic] = solver.solve(b);
    }


    auto* Part1data = (unsigned char*)malloc(target.width * target.hight * target.n);
    auto* Part2data = (unsigned char*)malloc(restore_target.width * restore_target.hight * restore_target.n);

    for (int j = 0; j < target.hight; j++) {
        for (int i = 0; i < target.width; i++) {
            Part1data[i * 3 + 0 + j * target.width * 3] = target.idata[i * 3 + 0 + j * 3 * target.width];
            Part1data[i * 3 + 1 + j * target.width * 3] = target.idata[i * 3 + 1 + j * 3 * target.width];
            Part1data[i * 3 + 2 + j * target.width * 3] = target.idata[i * 3 + 2 + j * 3 * target.width];
        }
    }
    

    if (Poisson) {
        for (int j = 0; j < restore_target.hight; j++) {
            for (int i = 0; i < restore_target.width; i++) {
                Part2data[i * 3 + 0 + j * restore_target.width * 3] = restore_target.idata[i * 3 + 0 + j * 3 * restore_target.width];
                Part2data[i * 3 + 1 + j * restore_target.width * 3] = restore_target.idata[i * 3 + 1 + j * 3 * restore_target.width];
                Part2data[i * 3 + 2 + j * restore_target.width * 3] = restore_target.idata[i * 3 + 2 + j * 3 * restore_target.width];
            }
        }
        Poisson_restore(restore_target, Part2data);
        string outputPath = "out.bmp";
        // write
        stbi_write_png(outputPath.c_str(), restore_target.width, restore_target.hight, restore_target.n, Part2data, 0);
        stbi_image_free(restore_target.idata);
        stbi_image_free(Part2data);
    }
    else
    {
        for (int j = 0; j < source_mask.hight; j++) {
            for (int i = 0; i < source_mask.width; i++) {
                if (isMaskPixel(i, j, source_mask)) {
                    unsigned int offset_position = mx * 3 + my * target.width * 3;
                    unsigned int k = varMap[maskFlatten(i, j, source_mask.width)];
                    vec3 col = vec3((float)solutionChannels[0][k], (float)solutionChannels[1][k], (float)solutionChannels[2][k]);

                    col[0] = clamp(col[0]);
                    col[1] = clamp(col[1]);
                    col[2] = clamp(col[2]);

                    Part1data[offset_position + i * 3 + 0 + j * target.width * 3] = (unsigned char)(col[0] * 255.0);
                    Part1data[offset_position + i * 3 + 1 + j * target.width * 3] = (unsigned char)(col[1] * 255.0);
                    Part1data[offset_position + i * 3 + 2 + j * target.width * 3] = (unsigned char)(col[2] * 255.0);
                }
            }
        }

        string outputPath = "out.bmp";
        // write
        stbi_write_png(outputPath.c_str(), target.width, target.hight, target.n, Part1data, 0);
        stbi_image_free(target.idata);
        stbi_image_free(Part1data);
    }
    //cin.get();
   return 0;
}