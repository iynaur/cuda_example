#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <string.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

#include <SDL/SDL.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

// Number of iterations per pixel
#define ITERA 120

#define THREADS_PER_BLOCK   512

int MAXX;
int MAXY;

GLuint buffer;
GLuint tex;

#  define CU_SAFE_CALL_NO_SYNC( call ) {                                     \
    cudaError_t err = call;                                                  \
    if( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CU_SAFE_CALL( call )       CU_SAFE_CALL_NO_SYNC(call);

unsigned __constant__ lookup[256];

int readtime()
{
    return SDL_GetTicks();
}

int kbhit(int *xx, int *yy)
{
    SDL_PumpEvents();

    Uint8 *keystate = SDL_GetKeyState(NULL);
    if ( keystate[SDLK_ESCAPE] )
    return 1;

    int x,y;
    Uint8 btn = SDL_GetMouseState (&x, &y);
    if (btn & SDL_BUTTON(SDL_BUTTON_LEFT)) {
    *xx = x;
    *yy = y;
    return 2;
    }
    if (btn & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
    *xx = x;
    *yy = y;
    return 3;
    }
    return 0;
}

void create_buffer(GLuint* buffer)
{
    glGenBuffersARB(1, buffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, *buffer);
    glBufferData(GL_PIXEL_PACK_BUFFER, MAXX*MAXY*4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void destroy_buffer(GLuint* buffer)
{
    glBindBuffer(GL_TEXTURE_2D, 0);
    glDeleteBuffers(1, buffer);
    *buffer = 0;
}

void create_texture(GLuint* tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, MAXX, MAXY, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void destroy_texture(GLuint* tex)
{
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, tex);
}

typedef union colorTag {
    struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
    };
    unsigned value;
} color;

void setPalette()
{
    color palette[256];

    int i;
    int ofs=0;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 16*(16-abs(i-16));
    palette[i+ofs].g = 0;
    palette[i+ofs].b = 16*abs(i-16);
    }
    ofs= 16;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 0;
    palette[i+ofs].g = 16*(16-abs(i-16));
    palette[i+ofs].b = 0;
    }
    ofs= 32;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 0;
    palette[i+ofs].g = 0;
    palette[i+ofs].b = 16*(16-abs(i-16));
    }
    ofs= 48;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 16*(16-abs(i-16));
    palette[i+ofs].g = 16*(16-abs(i-16));
    palette[i+ofs].b = 0;
    }
    ofs= 64;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 0;
    palette[i+ofs].g = 16*(16-abs(i-16));
    palette[i+ofs].b = 16*(16-abs(i-16));
    }
    ofs= 80;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 16*(16-abs(i-16));
    palette[i+ofs].g = 0;
    palette[i+ofs].b = 16*(16-abs(i-16));
    }
    ofs= 96;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 16*(16-abs(i-16));
    palette[i+ofs].g = 16*(16-abs(i-16));
    palette[i+ofs].b = 16*(16-abs(i-16));
    }
    ofs= 112;
    for (i = 0; i < 16; i++) {
    palette[i+ofs].r = 16*(8-abs(i-8));
    palette[i+ofs].g = 16*(8-abs(i-8));
    palette[i+ofs].b = 16*(8-abs(i-8));
    }
    CU_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol(
        lookup,
    palette,
    256 * sizeof(unsigned)) );
}

__global__ void CoreLoop(int *p, float xld, float yld, float xru, float yru, int MAXX, int MAXY)
{
    float re,im,rez,imz;
    float t1, t2, o1, o2;
    int k;
    unsigned result = 0;
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    int y = idx / MAXX;
    int x = idx % MAXX;

    re = (float) xld + (xru-xld)*x/MAXX;
    im = (float) yld + (yru-yld)*y/MAXY;

    rez = 0.0f;
    imz = 0.0f;

    k = 0;
    while (k < ITERA) {
    o1 = rez * rez;
    o2 = imz * imz;
    t2 = 2 * rez * imz;
    t1 = o1 - o2;
    rez = t1 + re;
    imz = t2 + im;
    if (o1 + o2 > 4) {
        result = k;
        break;
    }
    k++;
    }

    p[y*MAXX + x] = lookup[result];
}

void mandel(double xld, double yld, double xru, double yru)
{
    int blocks = MAXX*MAXY / THREADS_PER_BLOCK;
    int *pixels;

    glBindTexture(GL_TEXTURE_2D, tex);
    CU_SAFE_CALL(cudaGLMapBufferObject((void**)&pixels, buffer));

    CoreLoop<<< blocks, THREADS_PER_BLOCK >>>(pixels, xld, yld, xru, yru, MAXX, MAXY);
    //CU_SAFE_CALL(cudaThreadSynchronize());
    CU_SAFE_CALL(cudaGLUnmapBufferObject(buffer));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MAXX, MAXY, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();

    SDL_GL_SwapBuffers();
}

int main(int argc, char *argv[])
{
    int st, en;

    switch (argc) {
    case 3:
        MAXX = atoi(argv[1]);
        MAXY = atoi(argv[2]);
    MAXX = 16*(MAXX/16);
    MAXY = 16*(MAXY/16);
        break;
    default:
        MAXX = 800;
        MAXY = 600;
        break;
    }

    printf("\nMandelbrot Zoomer by Thanassis (an experiment in CUDA).\n\n");
    printf("Stats:\n\t");
    printf("(CUDA calculation - use left and right mouse buttons to zoom in/out)\n\t");

    if ( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        printf("Couldn't initialize SDL: %d\n", SDL_GetError());
        exit(1);
    }

    // Clean up on exit
    atexit(SDL_Quit);

    if (!SDL_SetVideoMode( MAXX, MAXY, 0, SDL_OPENGL)) {
        printf("Couldn't set video mode: %d\n", SDL_GetError());
        exit(1);
    }

    SDL_EventState(SDL_MOUSEMOTION, SDL_IGNORE);

    glViewport(0, 0, MAXX, MAXY);
    glClearColor(0.3f, 0.3f, 0.3f, 0.5f);         // This Will Clear The Background Color To Black
    glEnable(GL_TEXTURE_2D);
    glLoadIdentity();

    setPalette();

    create_buffer(&buffer);
    create_texture(&tex);
    CU_SAFE_CALL(cudaGLRegisterBufferObject(buffer));

    st = readtime();
    int x;
    int y;

    double xld = -2., yld=-1.1, xru=-2+(MAXX/MAXY)*3., yru=1.1;
    unsigned i = 0;
    while(1) {
    mandel(xld, yld, xru, yru);
        int result = kbhit(&x, &y);
    if (result == 1)
            break;
    else if (result == 2 || result == 3) {
        double ratiox = double(x)/MAXX;
        double ratioy = double(y)/MAXY;
        double xrange = xru-xld;
        double yrange = yru-yld;
        double direction = result==2?1.:-1.;
        xld += direction*0.01*ratiox*xrange;
        xru -= direction*0.01*(1.-ratiox)*xrange;
        yld += direction*0.01*(1.-ratioy)*yrange;
        yru -= direction*0.01*ratioy*yrange;
    }
    i++;
    }
    en = readtime();

    CU_SAFE_CALL(cudaGLUnregisterBufferObject(buffer));
    destroy_texture(&tex);
    destroy_buffer(&buffer);

    printf("frames/sec:%5.2f\n\n", ((float) i) / ((en - st) / 1000.0f));

    return 0;
}
