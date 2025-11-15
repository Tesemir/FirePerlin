🔥 Advanced Fire Simulation with Multi-Noise Turbulence + ImGui (OpenGL / C++)

A real-time GPU particle-based fire simulation using OpenGL 3.3, supporting multiple turbulence models (Perlin, Simplex, Value, Gaussian, Pink Noise, Lagrangian advection, Navier-Stokes).
The project includes an interactive ImGui UI to tweak parameters live: emission rate, turbulence strength, noise scale, method type, etc.

This project uses:

OpenGL 3.3 Core

GLFW / GLAD

GLM

Dear ImGui

Custom noise generators

A lightweight 2D fluid solver (Navier-Stokes)

📸 Preview

(Add your screenshots here)

screenshots/
    fire_demo_1.png
    fire_demo_2.png

✨ Features
🔥 Fire Particle System

Up to thousands of GPU particles

Z-sorting for additive blending

Dynamic size, rotation & color transitions

Two particle types (embers + smoke-like puffs)

🌪 Turbulence Methods

Selectable via ImGui:

Method	Description
None	No turbulence
Perlin Noise	Classic Perlin, 3D sampling
Simplex Noise	More efficient smooth noise
Value Noise	Grid-based random values
Gaussian Noise	Random normal disturbution
Pink Noise (1/f)	Smooth low-frequency turbulence
Lagrangian	Curl-based semi-fluid velocity sampling
Navier-Stokes	Real 2D fluid solver used as a velocity field
📐 3D Grid

Rendered under the fire for spatial reference

Optional color highlight near fire position

🖥 ImGui Control Panel

Switch turbulence model

Adjust:

Noise scale

Turbulence strength

Emission rate

Pause / Resume

Reset Fluid solver

FPS counter window

Optional ImGui demo window

📦 Project Structure
project/
│
├── src/
│   ├── main.cpp          
│   ├── glad.c
│   └── shaders/         
│
├── dependencies/
│   ├── imgui/
│   │   ├── imgui.cpp
│   │   ├── backends/imgui_impl_glfw.cpp
│   │   └── backends/imgui_impl_opengl3.cpp
│   └── glfw / glad / glm if included manually
│
├── CMakeLists.txt
├── config.h
└── README.md

🔧 Build Instructions (CMake)
Requirements
Dependency	Version
CMake	3.26+
OpenGL	3.3 Core
GLFW	3.x
GLAD	Core Loader
GLM	0.9.9+
ImGui	Latest
Clone
git clone https://github.com/yourname/fire-simulation.git
cd fire-simulation

Build
mkdir build
cd build
cmake ..
cmake --build .

Run
./perlin_noise

🎮 Controls
Action	Key
Move forward	W
Move backward	S
Move left	A
Move right	D
Move up	Space
Move down	Left Shift
Mouse Look	Move mouse
Exit	ESC
🛠 How the System Works
Particle Buffers

Particles stored in a single VBO:

position (vec3)
color    (vec4)
size     (float)
rotation (float)


Uploaded each frame via:

glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());

Turbulence Sampling

Selectable via enum:

enum class TurbulenceMethod {
    None, Perlin, Simplex, Value, Gaussian, Pink, Lagrangian, NavierStokes
};


Each turbulence model returns a 3D vector force:

glm::vec3 turb = sampleTurbulenceVec(p.position, time);
p.velocity += turb * dt;

Navier-Stokes Integration

When active, the 2D fluid grid is updated every frame:

globalFluid.addDensity(...)
globalFluid.addVelocity(...)
globalFluid.step();


Then sampled as:

glm::vec2 v = globalFluid.sampleVelocity(pos.x, pos.z, 30.0f);

🧪 ImGui Panel

Your UI panel includes:

Turbulence method dropdown

Noise scale slider

Turbulence strength slider

Emission rate slider

Pause/Resume button

Reset fluid solver button

Debug windows

Code:

ImGui::Combo("Method", &cur, methods, IM_ARRAYSIZE(methods));
ImGui::SliderFloat("Noise scale", &noiseScale, 0.01f, 2.0f);
ImGui::SliderFloat("Turbulence strength", &turbStrength, 0.0f, 8.0f);
ImGui::SliderFloat("Emission rate", &emission, 0.0f, 2000.0f);
