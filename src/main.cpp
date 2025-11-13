#include "config.h"
#include "FluidGrid.h"
// Camera variables
glm::vec3 cameraPos = glm::vec3(0.0f, 2.0f, 5.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw = -90.0f;
float pitch = -10.0f;
float lastX = 640.0f;
float lastY = 360.0f;
bool firstMouse = true;

// Vertex Shader
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in float aSize;

out vec4 particleColor;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = max(aSize, 10.0);
    particleColor = aColor;
}
)";

// Fragment Shader
const char* fragmentShaderSource = R"(
#version 330 core
in vec4 particleColor;
out vec4 FragColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if(dist > 0.5)
        discard;
    
    float alpha = particleColor.a * (1.0 - dist * 2.0);
    FragColor = vec4(particleColor.rgb, alpha);
}
)";

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec4 color;
    float size;
    float life;
    float maxLife;
};


struct LagrangianParams {
    float buoyancy = 3.0f;
    float drag = 0.9f;
    float vorticityStrength = 0.5f;
    glm::vec3 globalWind = glm::vec3(0.0f);
};

class LagrangianEngine {
public:
    LagrangianParams params;
    LagrangianEngine() = default;

    void applyForces(Particle& p, float dt){
        // buoyancy: push particles upward proportional to heat (life/age) or density
        float heat = 1.0f - (p.life / p.maxLife);
        glm::vec3 buoy = glm::vec3(0.0f, params.buoyancy * heat, 0.0f);

        // drag
        p.velocity *= std::pow(params.drag, dt*60.0f);

        // wind
        p.velocity += params.globalWind * dt;

        // vorticity confinement (approx) - small curl creation from lateral velocity:
        glm::vec3 vorticity = glm::cross(p.velocity, glm::vec3(0.0f,1.0f,0.0f)); // cheap approx
        p.velocity += vorticity * params.vorticityStrength * dt;

        p.velocity += buoy * dt;
    }

    void step(std::vector<Particle>& particles, float dt){
        for(auto& p : particles){
            applyForces(p, dt);
            p.position += p.velocity * dt;
        }
    }
};


class FireSystem {
private:
    std::vector<Particle> particles;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
public:
    FireSystem(int numParticles) : dist(-1.0f, 1.0f) {
        rng.seed(std::random_device{}());
        particles.resize(numParticles);
        for(auto& p : particles) {
            resetParticle(p);
        }
    }
    
    void resetParticle(Particle& p) {
        float angle = dist(rng) * 3.14159f;
        float radius = std::abs(dist(rng)) * 0.5f;
        
        p.position = glm::vec3(
            cos(angle) * radius,
            0.0f,
            sin(angle) * radius
        );
        
        float swirl = angle + dist(rng) * 0.5f;
        p.velocity = glm::vec3(
            cos(swirl) * 0.4f,
            2.5f + dist(rng) * 1.0f,
            sin(swirl) * 0.4f
        );
        
        p.life = 0.0f;
        p.maxLife = 1.5f + std::abs(dist(rng)) * 1.0f;
        p.size = 50.0f + std::abs(dist(rng)) * 50.0f;
        p.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    
    void update(float dt) {
        for(auto& p : particles) {
            p.life += dt;
            
            if(p.life >= p.maxLife) {
                resetParticle(p);
            }
            
            p.position += p.velocity * dt;
            
            float turbulence = sin(p.life * 3.0f) * 0.8f;
            p.velocity.x += dist(rng) * turbulence * dt;
            p.velocity.z += dist(rng) * turbulence * dt;
            
            float angle = atan2(p.velocity.z, p.velocity.x);
            angle += 1.0f * dt;
            float speed = sqrt(p.velocity.x * p.velocity.x + p.velocity.z * p.velocity.z);
            p.velocity.x = cos(angle) * speed;
            p.velocity.z = sin(angle) * speed;
            
            p.velocity.y -= 0.8f * dt;
            p.velocity *= 0.98f;
            
            float t = p.life / p.maxLife;
            
            if(t < 0.2f) {
                float mix = t / 0.2f;
                p.color = glm::vec4(1.0f, 1.0f, 1.0f - mix * 0.3f, 1.0f);
            } else if(t < 0.4f) {
                float mix = (t - 0.2f) / 0.2f;
                p.color = glm::vec4(1.0f, 0.9f - mix * 0.2f, 0.1f, 1.0f);
            } else if(t < 0.6f) {
                float mix = (t - 0.4f) / 0.2f;
                p.color = glm::vec4(1.0f, 0.6f - mix * 0.2f, 0.0f, 1.0f - mix * 0.2f);
            } else if(t < 0.8f) {
                float mix = (t - 0.6f) / 0.2f;
                p.color = glm::vec4(0.9f, 0.3f - mix * 0.3f, 0.0f, 0.8f - mix * 0.3f);
            } else {
                float mix = (t - 0.8f) / 0.2f;
                p.color = glm::vec4(0.6f, 0.0f, 0.0f, 0.5f - mix * 0.5f);
            }
            
            float sizeLife = t < 0.3f ? t / 0.3f : 1.0f - (t - 0.3f) / 0.7f;
            p.size = (50.0f + std::abs(dist(rng)) * 50.0f) * sizeLife;
        }
    }
    
    const std::vector<Particle>& getParticles() const {
        return particles;
    }
};
enum class NoiseType {
    NONE,
    PERLIN,
    SIMPLEX,
    PINK,
    GAUSSIAN,
    VALUE,
    NAVIER_STOKES,
    LAGRANGIAN
};

class NoiseGenerator {
private:
    NoiseType noiseType;
    float noiseScale;
    float time;
public:
    NoiseGenerator(NoiseType type = NoiseType::PERLIN, float scale = 1.0f)
        : noiseType(type), noiseScale(scale), time(0.0f) {}

    void setNoiseType(NoiseType type) { noiseType = type; }
    void setScale(float scale) { noiseScale = scale; }
    NoiseType getNoise() {return noiseType;}

    glm::vec3 getNoiseOffset(const glm::vec3& pos, float dt) {
        time += dt;
        if (noiseType == NoiseType::NONE) return glm::vec3(0.0f);

        glm::vec3 offset(0.0f);
        glm::vec3 npos = pos * noiseScale + glm::vec3(0.0f, time * 0.2f, 0.0f);

        switch (noiseType) {
            case NoiseType::PERLIN:
            {
                offset.x = glm::perlin(glm::vec2(npos.x, npos.y)) * 0.5f;
                offset.y = glm::perlin(glm::vec2(npos.y, npos.z)) * 0.5f;
                offset.z = glm::perlin(glm::vec2(npos.z, npos.x)) * 0.5f;
                break;
            }

            case NoiseType::SIMPLEX:
            {
                offset.x = glm::perlin(glm::vec2(npos.x, npos.y) * 0.5f + glm::vec2(0.3f, 0.7f));
                offset.y = glm::perlin(glm::vec2(npos.y, npos.z) * 0.5f + glm::vec2(0.9f, 0.1f));
                offset.z = glm::perlin(glm::vec2(npos.z, npos.x) * 0.5f + glm::vec2(0.5f, 0.5f));
                break;
            }
            case NoiseType::VALUE: {
                auto valueNoise = [](float x, float y) {
                    int xi = static_cast<int>(std::floor(x)) & 255;
                    int yi = static_cast<int>(std::floor(y)) & 255;
                    float xf = x - std::floor(x);
                    float yf = y - std::floor(y);
                    float r = glm::fract(glm::sin(glm::dot(glm::vec2(xi, yi), glm::vec2(12.9898f, 78.233f))) * 43758.5453f);
                    return r;
                };
                offset.x = valueNoise(npos.x, npos.y) - 0.5f;
                offset.y = valueNoise(npos.y, npos.z) - 0.5f;
                offset.z = valueNoise(npos.z, npos.x) - 0.5f;
                break;
            }

            case NoiseType::GAUSSIAN: {
                static std::default_random_engine gen;
                static std::normal_distribution<float> dist(0.0f, 0.25f);
                offset.x = dist(gen);
                offset.y = dist(gen);
                offset.z = dist(gen);
                break;
            }

            case NoiseType::PINK: {
                static std::default_random_engine gen;
                static std::normal_distribution<float> dist(0.0f, 1.0f);
                float white = dist(gen);
                static float b0 = 0, b1 = 0, b2 = 0;
                b0 = 0.99765f * b0 + white * 0.0990460f;
                b1 = 0.96300f * b1 + white * 0.2965164f;
                b2 = 0.57000f * b2 + white * 1.0526913f;
                float pink = b0 + b1 + b2 + white * 0.1848f;
                offset = glm::vec3(pink * 0.1f);
                break;
            }
            
        }
        return offset;
    }
};

class FireSystemWithNoise : public FireSystem {
private:
    NoiseGenerator noiseGen;
public:
    FireSystemWithNoise(int numParticles, NoiseType noiseType = NoiseType::PERLIN, float scale = 1.0f)
        : FireSystem(numParticles), noiseGen(noiseType, scale) {}

    LagrangianEngine lagrange;
    FluidGrid* fluid = nullptr;

    glm::vec2 sampleGridVelocity(FluidGrid& g, float worldX, float worldY) {
        // Map world coords → grid space
        float gx = glm::clamp(worldX * g.N, 0.0f, (float)g.N - 1.001f);
        float gy = glm::clamp(worldY * g.N, 0.0f, (float)g.N - 1.001f);

        int i = static_cast<int>(floor(gx));
        int j = static_cast<int>(floor(gy));
        float s = gx - i;
        float t = gy - j;

        // Clamp neighbor indices (avoid i+1,j+1 going out of range)
        int i1 = glm::min(i + 1, g.N - 1);
        int j1 = glm::min(j + 1, g.N - 1);

        float vx00 = g.Vx[g.IX(i,  j)];
        float vx10 = g.Vx[g.IX(i1, j)];
        float vx01 = g.Vx[g.IX(i,  j1)];
        float vx11 = g.Vx[g.IX(i1, j1)];

        float vy00 = g.Vy[g.IX(i,  j)];
        float vy10 = g.Vy[g.IX(i1, j)];
        float vy01 = g.Vy[g.IX(i,  j1)];
        float vy11 = g.Vy[g.IX(i1, j1)];

        float vx = (1 - s) * (1 - t) * vx00 + s * (1 - t) * vx10 + (1 - s) * t * vx01 + s * t * vx11;
        float vy = (1 - s) * (1 - t) * vy00 + s * (1 - t) * vy10 + (1 - s) * t * vy01 + s * t * vy11;

        return glm::vec2(vx, vy);
    }


    void updateWithNoise(float dt) { 
        
        if(noiseGen.getNoise() == NoiseType::LAGRANGIAN)
        {
            auto& particles = const_cast<std::vector<Particle>&>(getParticles());
            lagrange.step(particles, dt);
        }
        else if(noiseGen.getNoise() == NoiseType::NAVIER_STOKES)
        {
            if (!fluid) return;
            int cx = fluid->N / 2;
            int cy = 2;
            for (int i = cx - 2; i <= cx + 2; i++) {
                for (int j = cy - 2; j <= cy + 2; j++) {
                    float strength = 2.0f - glm::length(glm::vec2(i - cx, j - cy)) * 0.3f;
                    if (strength > 0)
                        fluid->addVelocity(i, j, 0.0f, strength);
                }
            }

            fluid->step();

            auto& particles = const_cast<std::vector<Particle>&>(getParticles());
            for (auto& p : particles) {
                glm::vec2 vel2 = sampleGridVelocity(*fluid, p.position.x, p.position.z);
                p.velocity.x += vel2.x * 0.5f;
                p.velocity.z += vel2.y * 0.5f;
                p.position += p.velocity * dt;
            }
        }
        
        FireSystem::update(dt);

        auto& parts = const_cast<std::vector<Particle>&>(getParticles());
        for (auto& p : parts) {
            glm::vec3 noiseOffset = noiseGen.getNoiseOffset(p.position, dt);
            p.position += noiseOffset * dt * 10.0f;
        }
    }

    void setNoiseType(NoiseType t) { noiseGen.setNoiseType(t); }
    void setNoiseScale(float s) { noiseGen.setScale(s); }
    NoiseType getNoise() {return noiseGen.getNoise();}
};



void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if(firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;
    
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    
    yaw += xoffset;
    pitch += yoffset;
    
    if(pitch > 89.0f) pitch = 89.0f;
    if(pitch < -89.0f) pitch = -89.0f;
    
    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(direction);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void updateFire(GLFWwindow* window, FireSystemWithNoise& fire)
{   
    if(glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::PERLIN);
    }
    if(glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::SIMPLEX);
    }      
    if(glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::VALUE);
    }
    if(glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::GAUSSIAN);
    }    
    if(glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::PINK);
    }
    if(glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::LAGRANGIAN);
    }        
    if(glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::NAVIER_STOKES);
    }        
    if(glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
    {
        fire.setNoiseType(NoiseType::NONE);
    }        
    
}

void processInput(GLFWwindow* window, float deltaTime) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    float velocity = 3.0f * deltaTime;
    if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += velocity * cameraFront;
    if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= velocity * cameraFront;
    if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;
    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;
    if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += velocity * cameraUp;
    if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos -= velocity * cameraUp;
}

const char* noiseTypeToString(NoiseType type) {
    switch (type) {
        case NoiseType::NONE:     return "None";
        case NoiseType::PERLIN:   return "Perlin";
        case NoiseType::SIMPLEX:  return "Simplex";
        case NoiseType::PINK:     return "Pink";
        case NoiseType::GAUSSIAN: return "Gaussian";
        case NoiseType::VALUE:    return "Value";
        case NoiseType::LAGRANGIAN:    return "LAGRANGIAN";
        case NoiseType::NAVIER_STOKES:    return "NAVIER_STOKES";
        default:                  return "Unknown";
    }
}

int main() {
    if(!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "3D Fire - WASD + Mouse to move", NULL, NULL);
    if(!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);
    
    // Compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
    }
    
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
    }
    
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // FireSystem fire(600);
    FireSystemWithNoise fire(600, NoiseType::NONE, 0.8f);
    FluidGrid fluid(64, 0.0001f, 0.0001f);
    fire.fluid = &fluid;
    
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * 600, nullptr, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));
    glEnableVertexAttribArray(0);
    
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, color));
    glEnableVertexAttribArray(1);
    
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, size));
    glEnableVertexAttribArray(2);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    ImGui::StyleColorsDark();
    
    float lastFrame = 0.0f;
    
    std::cout << "\nControls:\n";
    std::cout << "  WASD - Move horizontally\n";
    std::cout << "  Mouse - Look around\n";
    std::cout << "  Space - Move up\n";
    std::cout << "  Shift - Move down\n";
    std::cout << "  ESC - Exit\n\n";
    
    while(!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        
        processInput(window, deltaTime);
        updateFire(window, fire);
        // fire.update(deltaTime);
        fire.updateWithNoise(deltaTime);
        
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        // glClearColor(0.1f, 0.1f, 0.05f, 1.0f);
        // glClearColor(1.0f, 1.0f, 1.0f, 0.5f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), 1280.0f / 720.0f, 0.1f, 100.0f);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_Always);

        ImGui::Begin("Controls", nullptr,
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::Text("Controls:");
        ImGui::Separator();
        ImGui::Text("[W][A][S][D] Move around");
        ImGui::Text("[Left Shift] Go down");
        ImGui::Text("[Space] Go Up");
        ImGui::Text("[1] Perlin Noise");
        ImGui::Text("[2] Simplex Noise");
        ImGui::Text("[3] Value Noise");
        ImGui::Text("[4] Guassian Noise");
        ImGui::Text("[5] Pink Noise");
        ImGui::Text("[6] Lagrangian Numerical");
        ImGui::Text("[7] Navier-Stoke Numerical");
        ImGui::Text("[0] No Noise");
        ImGui::Text("Now mode is: %s", noiseTypeToString(fire.getNoise()));
        ImGui::Text("[ESC] Exit");

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
        
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Particle) * fire.getParticles().size(), fire.getParticles().data());
        glDrawArrays(GL_POINTS, 0, fire.getParticles().size());

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    
    glfwTerminate();
    return 0;
}
