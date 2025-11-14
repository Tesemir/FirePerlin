// main.cpp
// Combined/extended fire effect with multiple noise methods + ImGui UI
// Requires: GLAD, GLFW, GLM, ImGui (and ImGui backends for GLFW+OpenGL3)

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>

// ------------------------ Existing Perlin implementation (unchanged) ------------------------
class PerlinNoise {
private:
    std::vector<int> p;
    double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    double lerp(double t, double a, double b) { return a + t * (b - a); }
    double grad(int hash, double x, double y, double z) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : h==12 || h==14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
public:
    PerlinNoise(unsigned int seed = 0) {
        p.resize(512);
        std::vector<int> permutation(256);
        for (int i = 0; i < 256; i++) permutation[i] = i;
        std::default_random_engine engine(seed);
        std::shuffle(permutation.begin(), permutation.end(), engine);
        for (int i = 0; i < 256; i++) p[i] = p[256 + i] = permutation[i];
    }
    double noise(double x, double y, double z) {
        int X = (int)floor(x) & 255;
        int Y = (int)floor(y) & 255;
        int Z = (int)floor(z) & 255;
        x -= floor(x); y -= floor(y); z -= floor(z);
        double u = fade(x), v = fade(y), w = fade(z);
        int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
        int B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
        return lerp(w,
                    lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x-1, y, z)),
                            lerp(u, grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z))),
                    lerp(v, lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)),
                            lerp(u, grad(p[AB+1], x, y-1, z-1), grad(p[BB+1], x-1, y-1, z-1))));
    }
};

// ------------------------ Simplex Noise (basic) ------------------------
// A compact simplex implementation (2D/3D). This is a basic version for demonstration.
class SimplexNoise {
private:
    std::vector<int> perm;
    static double dot(const int* g, double x, double y, double z) { return g[0]*x + g[1]*y + g[2]*z; }
    static const int grad3[12][3];
public:
    SimplexNoise(unsigned int seed = 0) {
        perm.resize(512);
        std::vector<int> p(256);
        for (int i=0;i<256;i++) p[i]=i;
        std::default_random_engine eng(seed);
        std::shuffle(p.begin(), p.end(), eng);
        for (int i=0;i<256;i++) perm[i]=perm[256+i]=p[i];
    }
    double noise(double xin, double yin, double zin) const {
        // Simplex noise 3D (classic)
        static const double F3 = 1.0/3.0;
        static const double G3 = 1.0/6.0;
        double s = (xin+yin+zin)*F3;
        int i = floor(xin+s), j=floor(yin+s), k=floor(zin+s);
        double t = (i+j+k)*G3;
        double X0 = i - t, Y0 = j - t, Z0 = k - t;
        double x0 = xin - X0, y0 = yin - Y0, z0 = zin - Z0;
        int i1,j1,k1, i2,j2,k2;
        if (x0>=y0) {
            if (y0>=z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
            else if (x0>=z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
            else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
        } else {
            if (y0<z0) { i1=0;j1=0;k1=1; i2=0;j2=1;k2=1; }
            else if (x0<z0) { i1=0;j1=1;k1=0; i2=0;j2=1;k2=1; }
            else { i1=0;j1=1;k1=0; i2=1;j2=1;k2=0; }
        }
        double x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
        double x2 = x0 - i2 + 2.0*G3, y2 = y0 - j2 + 2.0*G3, z2 = z0 - k2 + 2.0*G3;
        double x3 = x0 - 1.0 + 3.0*G3, y3 = y0 - 1.0 + 3.0*G3, z3 = z0 - 1.0 + 3.0*G3;
        int ii = i & 255, jj = j & 255, kk = k & 255;
        int gi0 = perm[ii+perm[jj+perm[kk]]] % 12;
        int gi1 = perm[ii+i1+perm[jj+j1+perm[kk+k1]]] % 12; // careful with typo - j1,k1, etc - define below
        // To avoid long messy index code, compute grad indices explicitly:
        int gi2 = perm[ii+i2+perm[jj+j2+perm[kk+k2]]] % 12;
        int gi3 = perm[ii+1+perm[jj+1+perm[kk+1]]] % 12;
        // compute contributions
        auto contrib = [&](double x, double y, double z, int gi) {
            double t = 0.6 - x*x - y*y - z*z;
            if (t<0) return 0.0;
            t *= t;
            return t * t * dot(grad3[gi], x, y, z);
        };
        return 32.0 * (contrib(x0,y0,z0,gi0) + contrib(x1,y1,z1,gi1) + contrib(x2,y2,z2,gi2) + contrib(x3,y3,z3,gi3));
    }
};
const int SimplexNoise::grad3[12][3] = {
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};

// ------------------------ Value noise ------------------------
class ValueNoise {
private:
    std::vector<float> values;
    int size;
    unsigned int seed;
    static float smoothstep(float t) { return t*t*(3-2*t); }
public:
    ValueNoise(int gridSize = 256, unsigned int seed_ = 0) : size(gridSize), seed(seed_) {
        values.resize((size+1)*(size+1));
        std::default_random_engine eng(seed);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int y=0;y<=size;y++) for (int x=0;x<=size;x++) values[y*(size+1)+x] = d(eng);
    }
    float sample(float x, float y, float z) const {
        // sample 2D value noise using x,z as coords, ignore y for lattice lookup (z used as second axis)
        float fx = x * 0.1f;
        float fz = z * 0.1f;
        int xi = floor(fx);
        int zi = floor(fz);
        float tx = fx - xi, tz = fz - zi;
        float sx = smoothstep(tx), sz = smoothstep(tz);
        auto getv = [&](int gx, int gz)->float {
            int ix = (gx % size + size) % size;
            int iz = (gz % size + size) % size;
            return values[iz*(size+1)+ix];
        };
        float v00 = getv(xi, zi);
        float v10 = getv(xi+1, zi);
        float v01 = getv(xi, zi+1);
        float v11 = getv(xi+1, zi+1);
        float a = v00 + (v10 - v00) * sx;
        float b = v01 + (v11 - v01) * sx;
        return a + (b - a) * sz;
    }
};

// ------------------------ Pink noise (Voss-McCartney approximation) ------------------------
class PinkNoise {
private:
    std::vector<float> rows;
    int maxRows;
    std::default_random_engine eng;
    std::uniform_real_distribution<float> uni;
public:
    PinkNoise(int rows_ = 16, unsigned int seed = 0) : maxRows(rows_), eng(seed), uni(-1.0f,1.0f) {
        rows.assign(maxRows, 0.0f);
        for (int i=0;i<maxRows;i++) rows[i] = uni(eng);
    }
    float next() {
        int i = rand() % maxRows;
        rows[i] = uni(eng);
        float sum=0; for (float v: rows) sum+=v;
        return sum / maxRows;
    }
};

// ------------------------ Utility RNGs ------------------------
std::mt19937 rng_engine((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
std::normal_distribution<float> normal_dist(0.0f, 1.0f);
std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

// ------------------------ Fluid solver (simple 2D stable fluids) ------------------------
struct FluidGrid {
    int N;
    float dt, diff, visc;
    std::vector<float> s, density;
    std::vector<float> Vx, Vy, Vx0, Vy0;
    FluidGrid(int N_ = 64, float diff_ = 0.0f, float visc_ = 0.0001f, float dt_ = 0.1f) : N(N_), dt(dt_), diff(diff_), visc(visc_) {
        s.assign((N+2)*(N+2), 0.0f);
        density.assign((N+2)*(N+2), 0.0f);
        Vx.assign((N+2)*(N+2), 0.0f);
        Vy.assign((N+2)*(N+2), 0.0f);
        Vx0.assign((N+2)*(N+2), 0.0f);
        Vy0.assign((N+2)*(N+2), 0.0f);
    }
    inline int IX(int x, int y) const { return x + (N+2)*y; }
    void addDensity(int x, int y, float amount) { density[IX(x,y)] += amount; }
    void addVelocity(int x, int y, float amountX, float amountY) { Vx[IX(x,y)] += amountX; Vy[IX(x,y)] += amountY; }

    // helper functions: lin_solve, diffuse, advect, project -- classic Stam
    void lin_solve(int b, std::vector<float>& x, const std::vector<float>& x0, float a, float c) {
        for (int k=0;k<20;k++) {
            for (int i=1;i<=N;i++) for (int j=1;j<=N;j++)
                x[IX(i,j)] = (x0[IX(i,j)] + a*( x[IX(i-1,j)] + x[IX(i+1,j)] + x[IX(i,j-1)] + x[IX(i,j+1)] )) / c;
            set_bnd(b, x);
        }
    }
    void diffuse(int b, std::vector<float>& x, const std::vector<float>& x0, float diff) {
        float a = dt * diff * N * N;
        lin_solve(b, x, x0, a, 1 + 4*a);
    }
    void advect(int b, std::vector<float>& d, const std::vector<float>& d0, const std::vector<float>& velocX, const std::vector<float>& velocY) {
        float dt0 = dt * N;
        for (int i=1;i<=N;i++) for (int j=1;j<=N;j++) {
            float x = i - dt0 * velocX[IX(i,j)];
            float y = j - dt0 * velocY[IX(i,j)];
            if (x<0.5f) x = 0.5f; if (x>N+0.5f) x = N+0.5f;
            int i0 = (int)floor(x), i1 = i0+1;
            if (y<0.5f) y = 0.5f; if (y>N+0.5f) y = N+0.5f;
            int j0 = (int)floor(y), j1 = j0+1;
            float s1 = x - i0, s0 = 1 - s1;
            float t1 = y - j0, t0 = 1 - t1;
            d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)] + t1*d0[IX(i0,j1)]) + s1*(t0*d0[IX(i1,j0)] + t1*d0[IX(i1,j1)]);
        }
        set_bnd(b, d);
    }
    void project(std::vector<float>& velocX, std::vector<float>& velocY, std::vector<float>& p, std::vector<float>& div) {
        for (int i=1;i<=N;i++) for (int j=1;j<=N;j++) {
            div[IX(i,j)] = -0.5f * (velocX[IX(i+1,j)] - velocX[IX(i-1,j)] + velocY[IX(i,j+1)] - velocY[IX(i,j-1)]) / N;
            p[IX(i,j)] = 0;
        }
        set_bnd(0, div); set_bnd(0, p);
        lin_solve(0, p, div, 1, 4);
        for (int i=1;i<=N;i++) for (int j=1;j<=N;j++) {
            velocX[IX(i,j)] -= 0.5f * (p[IX(i+1,j)] - p[IX(i-1,j)]) * N;
            velocY[IX(i,j)] -= 0.5f * (p[IX(i,j+1)] - p[IX(i,j-1)]) * N;
        }
        set_bnd(1, velocX); set_bnd(2, velocY);
    }
    void step() {
        diffuse(1, Vx0, Vx, visc); diffuse(2, Vy0, Vy, visc);
        project(Vx0, Vy0, Vx, Vy);
        advect(1, Vx, Vx0, Vx0, Vy0); advect(2, Vy, Vy0, Vx0, Vy0);
        project(Vx, Vy, Vx0, Vy0);
        diffuse(0, s, density, diff); advect(0, density, s, Vx, Vy);
    }
    void set_bnd(int b, std::vector<float>& x) {
        for (int i=1;i<=N;i++) {
            x[IX(0,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
            x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
            x[IX(i,0)] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
            x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
        }
        x[IX(0,0)] = 0.5f*(x[IX(1,0)] + x[IX(0,1)]);
        x[IX(0,N+1)] = 0.5f*(x[IX(1,N+1)] + x[IX(0,N)]);
        x[IX(N+1,0)] = 0.5f*(x[IX(N,0)] + x[IX(N+1,1)]);
        x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)] + x[IX(N+1,N)]);
    }
    // sample velocity at (x,z) in world coordinates; map to grid coordinates
    glm::vec2 sampleVelocity(float x, float z, float worldSize=30.0f) {
        float half = worldSize/2.0f;
        float fx = (x + half) / worldSize * N + 1;
        float fz = (z + half) / worldSize * N + 1;
        if (fx < 1) fx = 1; if (fx > N) fx = N;
        if (fz < 1) fz = 1; if (fz > N) fz = N;
        int i0 = (int)floor(fx), j0 = (int)floor(fz);
        int i1 = std::min(i0+1, N), j1 = std::min(j0+1, N);
        float sx = fx - i0, sy = fz - j0;
        float vx = (1-sx)*(1-sy)*Vx[IX(i0,j0)] + sx*(1-sy)*Vx[IX(i1,j0)] + (1-sx)*sy*Vx[IX(i0,j1)] + sx*sy*Vx[IX(i1,j1)];
        float vy = (1-sx)*(1-sy)*Vy[IX(i0,j0)] + sx*(1-sy)*Vy[IX(i1,j0)] + (1-sx)*sy*Vy[IX(i0,j1)] + sx*sy*Vy[IX(i1,j1)];
        return glm::vec2(vx, vy);
    }
};

// ------------------------ Particle & shaders (kept from your original) ------------------------
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec4 color;
    float size;
    float life;
    float maxLife;
    float rotation;
    int type;
};

// Vertex/fragment shaders (same as yours — kept minimal)
const char* particleVertexShader = R"(
// ... same as your original vertex shader ...
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in float aSize;
layout (location = 3) in float aRotation;
out vec4 vColor;
out float vRotation;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
void main() {
    vec4 viewPos = view * model * vec4(aPos, 1.0);
    gl_Position = projection * viewPos;
    float distance = length(viewPos.xyz);
    gl_PointSize = aSize * (400.0 / max(distance, 1.0));
    vColor = aColor;
    vRotation = aRotation;
}
)";

const char* particleFragmentShader = R"(
// ... same as your original fragment shader ...
#version 330 core
in vec4 vColor;
in float vRotation;
out vec4 FragColor;
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float s = sin(vRotation), c = cos(vRotation);
    mat2 rot = mat2(c, -s, s, c);
    coord = rot * coord;
    float dist = length(coord);
    if (dist > 0.5) discard;
    float alpha = pow(1.0 - dist * 2.0, 2.0);
    float noise = fract(sin(dot(coord * 20.0, vec2(12.9898, 78.233))) * 43758.5453);
    alpha *= 0.9 + 0.1 * noise;
    FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
)";

const char* gridVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 FragPos;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
void main() {
    FragPos = aPos;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* gridFragmentShader = R"(
#version 330 core
in vec3 FragPos;
out vec4 FragColor;
uniform vec3 lineColor;
uniform vec3 firePos;
void main() {
    float dist = length(FragPos.xz - firePos.xz);
    float brightness = 1.0 / (1.0 + dist * 0.3);
    vec3 color = lineColor + vec3(0.3, 0.2, 0.0) * brightness;
    FragColor = vec4(color, 1.0);
}
)";

// ------------------------ Global noise / systems ------------------------
PerlinNoise globalPerlin(12345);
SimplexNoise globalSimplex(1337);
ValueNoise globalValue(256, 4242);
PinkNoise globalPink(16, 999);
FluidGrid globalFluid(64, 0.0f, 0.0001f, 0.1f);

// ------------------------ Noise type enum ------------------------
enum class TurbulenceMethod {
    None = 0,
    Perlin,
    Simplex,
    Value,
    Gaussian,
    Pink,
    Lagrangian,  // uses chosen base noise to build velocity field and RK2 advect
    NavierStokes // sample from fluid grid
};

// ------------------------ FireEffect class (extended) ------------------------
class FireEffect {
private:
    std::vector<Particle> particles;
    GLuint particleVAO=0, particleVBO=0;
    GLuint gridVAO=0, gridVBO=0;
    GLuint particleShader=0, gridShader=0;
    const int MAX_PARTICLES = 15000;
    float emissionRate = 300.0f;
    float timeSinceLastEmission = 0.0f;
    float time = 0.0f;
    TurbulenceMethod method = TurbulenceMethod::Perlin;
    float noiseScale = 0.5f;
    float turbulenceStrength = 2.0f;
    bool paused = false;
public:
    FireEffect() {
        particles.reserve(MAX_PARTICLES);
        setupShaders();
        setupBuffers();
    }

    void setMethod(TurbulenceMethod m) { method = m; }
    TurbulenceMethod getMethod() const { return method; }
    void setNoiseScale(float s) { noiseScale = s; }
    void setTurbulenceStrength(float s) { turbulenceStrength = s; }
    void setEmissionRate(float r) { emissionRate = r; }
    void setPaused(bool p) { paused = p; }

    GLuint compileShader(const char* source, GLenum type) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        int success; glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[1024]; glGetShaderInfoLog(shader, 1024, nullptr, log);
            std::cout << "Shader compile error: " << log << std::endl;
        }
        return shader;
    }

    void setupShaders() {
        GLuint vs = compileShader(particleVertexShader, GL_VERTEX_SHADER);
        GLuint fs = compileShader(particleFragmentShader, GL_FRAGMENT_SHADER);
        particleShader = glCreateProgram();
        glAttachShader(particleShader, vs); glAttachShader(particleShader, fs);
        glLinkProgram(particleShader);
        glDeleteShader(vs); glDeleteShader(fs);

        vs = compileShader(gridVertexShader, GL_VERTEX_SHADER);
        fs = compileShader(gridFragmentShader, GL_FRAGMENT_SHADER);
        gridShader = glCreateProgram();
        glAttachShader(gridShader, vs); glAttachShader(gridShader, fs);
        glLinkProgram(gridShader);
        glDeleteShader(vs); glDeleteShader(fs);
    }

    void setupBuffers() {
        glGenVertexArrays(1, &particleVAO);
        glGenBuffers(1, &particleVBO);
        glBindVertexArray(particleVAO);
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, color)); glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, size)); glEnableVertexAttribArray(2);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, rotation)); glEnableVertexAttribArray(3);

        // grid
        std::vector<float> gridVertices;
        float gridSize = 30.0f;
        int gridLines = 60;
        float step = gridSize / gridLines;
        for (int i = 0; i <= gridLines; i++) {
            float p = -gridSize/2 + i * step;
            gridVertices.push_back(p); gridVertices.push_back(0.0f); gridVertices.push_back(-gridSize/2);
            gridVertices.push_back(p); gridVertices.push_back(0.0f); gridVertices.push_back(gridSize/2);
            gridVertices.push_back(-gridSize/2); gridVertices.push_back(0.0f); gridVertices.push_back(p);
            gridVertices.push_back(gridSize/2); gridVertices.push_back(0.0f); gridVertices.push_back(p);
        }
        glGenVertexArrays(1, &gridVAO);
        glGenBuffers(1, &gridVBO);
        glBindVertexArray(gridVAO);
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, gridVertices.size()*sizeof(float), gridVertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
    }

    void emitParticle() {
        if (particles.size() >= MAX_PARTICLES) return;
        Particle p;
        float radius = pow(uniform01(rng_engine), 2.0f) * 0.8f;
        float angle = uniform01(rng_engine) * 2.0f * 3.14159f;
        p.position = glm::vec3(radius * cos(angle), 0.05f + uniform01(rng_engine) * 0.1f, radius * sin(angle));
        float rand = uniform01(rng_engine);
        if (rand < 0.7f) {
            p.type = 0;
            p.velocity = glm::vec3(distribute(-0.4f,0.4f), 2.5f + uniform01(rng_engine) * 1.5f, distribute(-0.4f,0.4f));
            p.size = 30.0f + uniform01(rng_engine) * 50.0f;
            p.maxLife = 0.8f + uniform01(rng_engine) * 0.6f;
        } else {
            p.type = 1;
            p.velocity = glm::vec3(distribute(-0.8f,0.8f), 1.2f + uniform01(rng_engine) * 0.8f, distribute(-0.8f,0.8f));
            p.size = 50.0f + uniform01(rng_engine) * 80.0f;
            p.maxLife = 1.5f + uniform01(rng_engine) * 1.5f;
        }
        p.life = p.maxLife;
        p.color = glm::vec4(1.0f);
        p.rotation = uniform01(rng_engine) * 6.28f;
        particles.push_back(p);
    }

    static float distribute(float a, float b) {
        std::uniform_real_distribution<float> d(a,b);
        return d(rng_engine);
    }

    // sampling different turbulence sources
    glm::vec3 sampleTurbulenceVec(const glm::vec3& pos, float t) {
        switch (method) {
            case TurbulenceMethod::None: return glm::vec3(0.0f);
            case TurbulenceMethod::Perlin: {
                float nx = globalPerlin.noise(pos.x*noiseScale + t, pos.y*noiseScale, pos.z*noiseScale);
                float ny = globalPerlin.noise(pos.x*noiseScale, pos.y*noiseScale + t, pos.z*noiseScale);
                float nz = globalPerlin.noise(pos.x*noiseScale, pos.y*noiseScale, pos.z*noiseScale + t);
                return glm::vec3(nx, ny, nz) * turbulenceStrength;
            }
            case TurbulenceMethod::Simplex: {
                float nx = globalSimplex.noise(pos.x*noiseScale + t, pos.y*noiseScale, pos.z*noiseScale);
                float ny = globalSimplex.noise(pos.x*noiseScale, pos.y*noiseScale + t, pos.z*noiseScale);
                float nz = globalSimplex.noise(pos.x*noiseScale, pos.y*noiseScale, pos.z*noiseScale + t);
                return glm::vec3(nx, ny, nz) * turbulenceStrength;
            }
            case TurbulenceMethod::Value: {
                float nx = globalValue.sample(pos.x*noiseScale + t, pos.y*noiseScale, pos.z*noiseScale);
                float ny = globalValue.sample(pos.x*noiseScale, pos.y*noiseScale + t, pos.z*noiseScale);
                float nz = globalValue.sample(pos.x*noiseScale, pos.y*noiseScale, pos.z*noiseScale + t);
                return glm::vec3(nx, ny, nz) * turbulenceStrength;
            }
            case TurbulenceMethod::Gaussian: {
                // stateless gaussian: draw on the fly
                float gx = normal_dist(rng_engine);
                float gy = normal_dist(rng_engine);
                float gz = normal_dist(rng_engine);
                return glm::vec3(gx, gy, gz) * (turbulenceStrength * 0.8f);
            }
            case TurbulenceMethod::Pink: {
                float v = globalPink.next();
                // make a small 3D perturbation using sequential pink values
                float v2 = globalPink.next();
                float v3 = globalPink.next();
                return glm::vec3(v,v2,v3) * turbulenceStrength * 0.5f;
            }
            case TurbulenceMethod::Lagrangian: {
                // Build a velocity field from curl of noise (2D curl -> 3D vector) and integrate sample point with RK2 to return delta
                // We create a 2D velocity field from perlin/simplex and evaluate curl
                auto sampleBase = [&](const glm::vec3 &p, float offs)->glm::vec2 {
                    // choose base by alternating between perlin/simplex/value
                    float a = globalPerlin.noise(p.x*noiseScale + offs, p.y*noiseScale, p.z*noiseScale);
                    float b = globalSimplex.noise(p.x*noiseScale, p.y*noiseScale + offs, p.z*noiseScale);
                    return glm::vec2(a, b);
                };
                // RK2 advect small step (we use return as velocity perturbation)
                float h = 0.02f;
                glm::vec2 v1 = sampleBase(pos, t);
                glm::vec3 mid = pos + glm::vec3(v1.x, 0.0f, v1.y) * (h*0.5f);
                glm::vec2 v2 = sampleBase(mid, t+0.5f*h);
                glm::vec3 vel = glm::vec3((v1.x+v2.x)*0.5f, (v1.x+v2.x)*0.5f*0.3f, (v1.y+v2.y)*0.5f);
                return vel * turbulenceStrength;
            }
            case TurbulenceMethod::NavierStokes: {
                // sample 2D fluid grid and convert to 3D perturbation
                glm::vec2 v = globalFluid.sampleVelocity(pos.x, pos.z, 30.0f);
                return glm::vec3(v.x, 0.5f * v.y, v.y) * turbulenceStrength * 2.0f;
            }
        }
        return glm::vec3(0.0f);
    }

    void update(float dt) {
        if (paused) return;
        time += dt;
        timeSinceLastEmission += dt;
        while (timeSinceLastEmission > 1.0f / emissionRate) {
            emitParticle();
            timeSinceLastEmission -= 1.0f / emissionRate;
        }
        // advance fluid solver a bit if needed (add sources near fire)
        if (method == TurbulenceMethod::NavierStokes) {
            // inject some density and velocity near center
            int cx = globalFluid.N/2, cy = globalFluid.N/6;
            for (int i=-2;i<=2;i++) for (int j=-2;j<=2;j++) {
                int xi = std::clamp(cx+i,1,globalFluid.N);
                int yj = std::clamp(cy+j,1,globalFluid.N);
                globalFluid.addDensity(xi,yj, 50.0f * dt);
                globalFluid.addVelocity(xi,yj, 0.0f, 5.0f * dt);
            }
            globalFluid.step();
        }

        for (auto it = particles.begin(); it != particles.end();) {
            it->life -= dt;
            if (it->life <= 0.0f) { it = particles.erase(it); continue; }
            // turbulence sample
            glm::vec3 turb = sampleTurbulenceVec(it->position, time);
            // apply turbulence scaled by dt and particle type
            it->velocity += turb * dt;
            // gravity-ish lift decay
            it->velocity.y -= 0.5f * dt;
            it->position += it->velocity * dt;
            it->rotation += dt * (it->type == 1 ? 0.5f : 2.0f);
            float t = 1.0f - (it->life / it->maxLife);
            // color/size evolution (kept similar to original)
            if (it->type == 0) {
                if (t < 0.15f) {
                    float fade = t / 0.15f;
                    it->color = glm::vec4(1.0f,1.0f,1.0f, std::min(1.2f, fade * 1.2f));
                    it->size *= 1.0f + dt * 2.0f;
                } else if (t < 0.35f) {
                    float fade = (t - 0.15f) / 0.2f;
                    it->color = glm::vec4(1.0f, 1.0f - fade * 0.3f, 0.6f - fade * 0.4f, 1.2f);
                    it->size *= 1.0f + dt * 1.5f;
                } else if (t < 0.6f) {
                    float fade = (t - 0.35f) / 0.25f;
                    it->color = glm::vec4(1.0f, 0.7f - fade * 0.3f, 0.2f - fade * 0.15f, 1.1f);
                    it->size *= 1.0f + dt * 1.0f;
                } else {
                    float fade = (t - 0.6f) / 0.4f;
                    it->color = glm::vec4(0.9f - fade * 0.6f, 0.4f - fade * 0.3f, 0.05f - fade * 0.05f, (1.0f - fade) * 0.9f);
                    it->size *= 1.0f + dt * 0.5f;
                }
            } else {
                if (t < 0.3f) {
                    float fade = t / 0.3f;
                    it->color = glm::vec4(0.2f + fade * 0.2f, 0.1f + fade * 0.15f, 0.05f + fade * 0.1f, fade * 0.6f);
                } else {
                    float fade = (t - 0.3f) / 0.7f;
                    it->color = glm::vec4(0.4f - fade * 0.2f, 0.25f - fade * 0.15f, 0.15f - fade * 0.1f, 0.6f * (1.0f - fade));
                }
                it->size *= 1.0f + dt * 1.2f;
            }
            ++it;
        }
        // sort by z for additive blending
        std::sort(particles.begin(), particles.end(), [](const Particle& a, const Particle& b) {
            return a.position.z > b.position.z;
        });
    }

    void render(const glm::mat4& projection, const glm::mat4& view) {
        glm::mat4 model = glm::mat4(1.0f);
        glUseProgram(gridShader);
        glUniformMatrix4fv(glGetUniformLocation(gridShader,"projection"),1,GL_FALSE,glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(gridShader,"view"),1,GL_FALSE,glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(gridShader,"model"),1,GL_FALSE,glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(gridShader,"lineColor"), 0.0f, 0.6f, 0.8f);
        glUniform3f(glGetUniformLocation(gridShader,"firePos"), 0.0f,0.0f,0.0f);
        glBindVertexArray(gridVAO);
        // count lines = (gridLines+1)*4 -> original had 244; just draw buffer size / 3 per vertex
        // we created  (gridLines+1)*4 vertices, but we used them as pairs for GL_LINES -> safe to draw many
        glDrawArrays(GL_LINES, 0, 244);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);

        glUseProgram(particleShader);
        glUniformMatrix4fv(glGetUniformLocation(particleShader,"projection"),1,GL_FALSE,glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(particleShader,"view"),1,GL_FALSE,glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(particleShader,"model"),1,GL_FALSE,glm::value_ptr(model));

        glBindVertexArray(particleVAO);
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());
        glDrawArrays(GL_POINTS, 0, (GLsizei)particles.size());

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }
};

// ------------------------ Camera & Input (kept from your original) ------------------------
class Camera {
public:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    float yaw, pitch, speed, sensitivity;
    Camera(glm::vec3 pos = glm::vec3(0.0f,4.0f,10.0f)) {
        position = pos; worldUp = glm::vec3(0,1,0);
        yaw = -90.0f; pitch = -20.0f; speed = 5.0f; sensitivity = 0.1f;
        updateVectors();
    }
    glm::mat4 getViewMatrix() { return glm::lookAt(position, position + front, up); }
    void processKeyboard(int dir, float dt) {
        float vel = speed * dt;
        if (dir==0) position += front * vel;
        if (dir==1) position -= front * vel;
        if (dir==2) position -= right * vel;
        if (dir==3) position += right * vel;
        if (dir==4) position += up * vel;
        if (dir==5) position -= up * vel;
    }
    void processMouseMovement(float xo, float yo) {
        xo *= sensitivity; yo *= sensitivity;
        yaw += xo; pitch += yo;
        if (pitch>89.0f) pitch=89.0f; if (pitch<-89.0f) pitch=-89.0f;
        updateVectors();
    }
private:
    void updateVectors() {
        glm::vec3 f;
        f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        f.y = sin(glm::radians(pitch));
        f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(f);
        right = glm::normalize(glm::cross(front, worldUp));
        up = glm::normalize(glm::cross(right, front));
    }
};

Camera camera;
float lastX = 700.0f, lastY = 450.0f; bool firstMouse = true;
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX; float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    camera.processMouseMovement(xoffset, yoffset);
}

// ------------------------ Main program ------------------------
int main() {
    if (!glfwInit()) { std::cout<<"GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow(1400,900,"Fire Effect - MultiNoise + ImGui", nullptr, nullptr);
    if (!window) { std::cout<<"Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window); glfwSwapInterval(1);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window, mouse_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cout<<"GLAD init failed\n"; return -1; }
    glEnable(GL_DEPTH_TEST); glEnable(GL_PROGRAM_POINT_SIZE); glEnable(GL_MULTISAMPLE);

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    FireEffect fire;

    float lastFrame = 0.0f;
    bool show_demo = false;
    bool paused = false;
    TurbulenceMethod selMethod = TurbulenceMethod::Perlin;
    float noiseScale = 0.5f;
    float turbStrength = 2.0f;
    float emission = 300.0f;

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame; lastFrame = currentFrame;

        // Input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.processKeyboard(0, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.processKeyboard(1, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.processKeyboard(2, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.processKeyboard(3, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.processKeyboard(4, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.processKeyboard(5, deltaTime);

        // ImGui new frame
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

        ImGui::Begin("Turbulence Controls");
        ImGui::Text("Choose turbulence method and parameters:");
        const char* methods[] = {"None","Perlin","Simplex","Value","Gaussian","Pink","Lagrangian","NavierStokes"};
        static int cur = 1;
        if (ImGui::Combo("Method", &cur, methods, IM_ARRAYSIZE(methods))) {
            selMethod = (TurbulenceMethod)cur;
            fire.setMethod(selMethod);
        }
        if (ImGui::SliderFloat("Noise scale", &noiseScale, 0.01f, 2.0f)) fire.setNoiseScale(noiseScale);
        if (ImGui::SliderFloat("Turbulence strength", &turbStrength, 0.0f, 8.0f)) fire.setTurbulenceStrength(turbStrength);
        if (ImGui::SliderFloat("Emission rate", &emission, 0.0f, 2000.0f)) fire.setEmissionRate(emission);
        if (ImGui::Button(paused ? "Resume" : "Pause")) { paused = !paused; fire.setPaused(paused); }
        if (ImGui::Button("Reset Fluid")) {
            // simple reset
            globalFluid = FluidGrid(globalFluid.N, globalFluid.diff, globalFluid.visc, globalFluid.dt);
        }
        ImGui::Checkbox("ImGui Demo",&show_demo);
        ImGui::End();

        ImGuiIO& io = ImGui::GetIO();
        ImVec2 window_pos = ImVec2(io.DisplaySize.x - 10.0f, 10.0f);
        ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f); 

        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);

        ImGui::Begin("FPS", nullptr,
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoTitleBar
        );

        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::End();

        if (show_demo) ImGui::ShowDemoWindow(&show_demo);

        // update
        fire.update(deltaTime);

        // render
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), 1400.0f/900.0f, 0.1f, 100.0f);
        glm::mat4 view = camera.getViewMatrix();
        fire.render(projection, view);

        // ImGui render
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window); glfwPollEvents();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}