#ifndef GRAPHICSMODE_H
#define GRAPHICSMODE_H
#define GL_H
#define GL_GLEXT_PROTOTYPES
#define GRAPHICS_H
#endif

#define cimg_display 0

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h> 
#include <helper_functions.h>
#include <CImg.h>
#include <omp.h>



using namespace std;
using namespace cimg_library;


#define W 512
#define H 786
#define TX 16
#define TY 16
#define RAD 1
#define GL_TEXTURE_TYPE GL_TEXTURE_2D

static const char *windname = "Sobel Edge Detector on GPU";

//Graphics Resource objects
GLuint pbo = 0;
GLuint tex = 0;
GLuint shader;
struct cudaGraphicsResource *cuda_pbo_resource;
struct uchar4;
struct float4;
unsigned int  *hImage  = NULL;



int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
    return idx > (idxMax-1) ? (idxMax-1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
    return idxClip(col, width) + idxClip(row, height)*width;
}

__global__
void sharpenKernel(uchar4 *d_out, const uchar4 *d_in,const float *d_filter, int w, int h) {
    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int r = threadIdx.y + blockDim.y * blockIdx.y;
    
    if ((c >= w) || (r >= h)) return;    
    
    const int i = flatten(c, r, w, h);
    const int fltSz = 2*RAD + 1;
    float rgb[3] = {0.f, 0.f, 0.f};

    for (int rd = -RAD; rd <= RAD; ++rd) {
        for (int cd = -RAD; cd <= RAD; ++cd) {
            int imgIdx = flatten(c + cd, r + rd, w, h);
            int fltIdx = flatten(RAD + cd, RAD + rd, fltSz, fltSz);
            uchar4 color = d_in[imgIdx];
            float weight = d_filter[fltIdx];
            rgb[0] += weight*color.x;
            rgb[1] += weight*color.y;
            rgb[2] += weight*color.z;
        }
    }    
    d_out[i].x = clip(rgb[0]);
    d_out[i].y = clip(rgb[1]);
    d_out[i].z = clip(rgb[2]);
}

// Cuda run model
 void cudarun() {
    //Image
    CImg<unsigned char> img ("src2.bmp");
    
    //input data
    uchar4 *arr=(uchar4*)malloc(W*H*sizeof(uchar4));
    
    // Copy data to array
    #pragma omp parallel 
    for (int r = 0; r < H; ++r) {
        #pragma omp parallel  for
        for (int c = 0; c < W; ++c){
            arr[r*W + c].x = img(c,r, 0);
            arr[r*W + c].y = img(c,r, 0);
            arr[r*W + c].z = img(c,r, 0);
            arr[r*W + c].w = 0;
        }
    }
    
    //device storage 
    const int fltSz = 2 * RAD + 1;
    const float filter[9] = {-1.0, -1.0, -1.0,
			-1.0, 8.0, -1.0,
			-1.0, -1.0, -1.0};    
    uchar4 *d_in = 0, *d_out = 0;
    float *d_filter = 0;
    
    cudaMalloc(&d_in, W*H*sizeof(uchar4));
    cudaMemcpy(d_in, arr, W*H*sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, W*H*sizeof(uchar4));
    cudaMalloc(&d_filter, fltSz*fltSz*sizeof(float));
    cudaMemcpy(d_filter, filter, fltSz*fltSz*sizeof(float),cudaMemcpyHostToDevice);    
       
    //Graphics resources map
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,cuda_pbo_resource);    
    
    //kernelLauncher   
    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(W, blockSize.x), divUp(H, blockSize.y)); 
    
    sharpenKernel<<<gridSize, blockSize>>>(d_out, d_in, d_filter, W, H);  
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

 }
 
// Display model
static void display(){
    //run kernel block
    cudarun();
    {   
        glClearColor (0.0, 0.0, 0.0, 0.0);
        glEnable(GL_DEPTH_TEST);
        glShadeModel(GL_SMOOTH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        //Textue map   
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);   
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
              
        
        glPushMatrix();
        glBegin(GL_QUADS);
    	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, (GLfloat)H);
    	glTexCoord2f(1.0f, 1.0f); glVertex2f((GLfloat)W, (GLfloat)H);
    	glTexCoord2f(1.0f, 0.0f); glVertex2f((GLfloat)W, 0.0f);
        glEnd();
        glPopMatrix();
        
        glActiveTexture(0);
        glBindTexture(GL_TEXTURE_TYPE, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST); 
       }
    glFlush();
    glutSwapBuffers();      
}

// Reshape window
static void reshape(int w, int h){
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, W, H, 0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);
    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    return program_id;
}

// Pixel Buffer generate
static void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 4 * W * H * sizeof(GLubyte), 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 1);
    
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(0);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,cudaGraphicsMapFlagsWriteDiscard);
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

// Free Buffer and Texture
static void exitfunc() {
    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
        glDeleteProgramsARB(1, &shader);
    }
}

// Handler for key event
static void keyboard(unsigned char key, int x, int y) {
  if(x==0||y==0) return;
    switch (key){
        case (27) :
            if (key==27||key=='q'||key=='Q')
                exit(EXIT_SUCCESS);
            else
                glutDestroyWindow(glutGetWindow());
                return; 
        default:
            break;        
    }
}

//Main block

int main(int argc, char** argv) {
    int dev=0;
    int runtimeVersion = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stderr,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    fprintf(stderr,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);    
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE| GLUT_DEPTH);
    glutInitWindowSize(W, H);
    glutInitWindowPosition(200, 200);
    glutCreateWindow(windname);     
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    initPixelBuffer();
    glutKeyboardFunc(keyboard);
    glutMainLoop();
    atexit(exitfunc);
    return EXIT_SUCCESS;
}