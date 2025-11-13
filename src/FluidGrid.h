// FluidGrid.h (drop next to your other sources)
#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

class FluidGrid {
public:
    int N;            // cells per side
    float dt;
    float diff;       // viscosity
    float visc;
    std::vector<float> s;   // temporary density
    std::vector<float> density;
    std::vector<float> Vx, Vy;       // velocity
    std::vector<float> Vx0, Vy0;     // prev velocity

    FluidGrid(int N_, float diffusion = 0.0f, float viscosity = 0.0001f, float dt_ = 0.016f)
        : N(N_), dt(dt_), diff(diffusion), visc(viscosity)
    {
        int size = (N+2)*(N+2);
        s.assign(size, 0.0f);
        density.assign(size, 0.0f);
        Vx.assign(size, 0.0f);
        Vy.assign(size, 0.0f);
        Vx0.assign(size, 0.0f);
        Vy0.assign(size, 0.0f);
    }

    inline int IX(int i,int j){ return i + (N+2)*j; }

    void addDensity(int x,int y,float amount){ density[IX(x,y)] += amount; }
    void addVelocity(int x,int y,float amountX,float amountY){
        int idx = IX(x,y);
        Vx[idx] += amountX; Vy[idx] += amountY;
    }

    void step(){
        diffuse(1, Vx0, Vx, visc, dt);
        diffuse(2, Vy0, Vy, visc, dt);

        project(Vx0, Vy0, Vx, Vy);

        advect(1, Vx, Vx0, Vx0, Vy0, dt);
        advect(2, Vy, Vy0, Vx0, Vy0, dt);

        project(Vx, Vy, Vx0, Vy0);

        diffuse(0, s, density, diff, dt);
        advect(0, density, s, Vx, Vy, dt);
    }

private:
    void set_bnd(int b, std::vector<float>& x){
        for(int i=1;i<=N;i++){
            x[IX(0,i)]   = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
            x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
            x[IX(i,0)]   = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
            x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
        }
        x[IX(0,0)] = 0.5f*(x[IX(1,0)] + x[IX(0,1)]);
        x[IX(0,N+1)] = 0.5f*(x[IX(1,N+1)] + x[IX(0,N)]);
        x[IX(N+1,0)] = 0.5f*(x[IX(N,0)] + x[IX(N+1,1)]);
        x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)] + x[IX(N+1,N)]);
    }

    void lin_solve(int b, std::vector<float>& x, const std::vector<float>& x0, float a, float c){
        for(int k=0;k<20;k++){
            for(int i=1;i<=N;i++){
                for(int j=1;j<=N;j++){
                    x[IX(i,j)] = (x0[IX(i,j)] + a*( x[IX(i-1,j)] + x[IX(i+1,j)] + x[IX(i,j-1)] + x[IX(i,j+1)] )) / c;
                }
            }
            set_bnd(b, x);
        }
    }

    void diffuse(int b, std::vector<float>& x, const std::vector<float>& x0, float diff, float dt){
        float a = dt * diff * N * N;
        lin_solve(b, x, x0, a, 1 + 4*a);
    }

    void advect(int b, std::vector<float>& d, const std::vector<float>& d0, const std::vector<float>& velocX, const std::vector<float>& velocY, float dt){
        float dt0 = dt * N;
        for(int i=1;i<=N;i++){
            for(int j=1;j<=N;j++){
                float x = i - dt0 * velocX[IX(i,j)];
                float y = j - dt0 * velocY[IX(i,j)];
                if(x < 0.5f) x = 0.5f;
                if(x > N + 0.5f) x = N + 0.5f;
                int i0 = (int)std::floor(x);
                int i1 = i0 + 1;
                if(y < 0.5f) y = 0.5f;
                if(y > N + 0.5f) y = N + 0.5f;
                int j0 = (int)std::floor(y);
                int j1 = j0 + 1;
                float s1 = x - i0;
                float s0 = 1 - s1;
                float t1 = y - j0;
                float t0 = 1 - t1;
                d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)] + t1*d0[IX(i0,j1)]) +
                             s1*(t0*d0[IX(i1,j0)] + t1*d0[IX(i1,j1)]);
            }
        }
        set_bnd(b, d);
    }

    void project(std::vector<float>& velocX, std::vector<float>& velocY, std::vector<float>& p, std::vector<float>& div){
        for(int i=1;i<=N;i++){
            for(int j=1;j<=N;j++){
                div[IX(i,j)] = -0.5f*(velocX[IX(i+1,j)] - velocX[IX(i-1,j)] + velocY[IX(i,j+1)] - velocY[IX(i,j-1)])/N;
                p[IX(i,j)] = 0;
            }
        }
        set_bnd(0, div);
        set_bnd(0, p);
        lin_solve(0, p, div, 1, 4);

        for(int i=1;i<=N;i++){
            for(int j=1;j<=N;j++){
                velocX[IX(i,j)] -= 0.5f * (p[IX(i+1,j)] - p[IX(i-1,j)]) * N;
                velocY[IX(i,j)] -= 0.5f * (p[IX(i,j+1)] - p[IX(i,j-1)]) * N;
            }
        }
        set_bnd(1, velocX);
        set_bnd(2, velocY);
    }

    // overload using internal buffers
    void project(std::vector<float>& velocX, std::vector<float>& velocY){
        project(velocX, velocY, Vx0, Vy0);
    }
};
