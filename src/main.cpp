#include "config.h"

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
    
    FireSystem fire(600);
    
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
        fire.update(deltaTime);
        
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), 1280.0f / 720.0f, 0.1f, 100.0f);
        
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Particle) * fire.getParticles().size(), fire.getParticles().data());
        glDrawArrays(GL_POINTS, 0, fire.getParticles().size());
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    
    glfwTerminate();
    return 0;
}