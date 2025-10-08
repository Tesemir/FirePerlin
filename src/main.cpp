#include "config.h"

#define M_PI 3.14159265358979323846

const char* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "layout (location = 1) in vec3 aColor;\n"
    "out vec3 ourColor;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "   ourColor = aColor;\n"
    "}\0";

const char* fragmentShaderSource = "#version 330 core\n"
    "in vec3 ourColor;\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(ourColor, 1.0);\n"
    "}\n";

void generateEllipse(float* vertices, int segments, float cx, float cy, float rx, float ry, float r, float g, float b) {
    vertices[0] = cx; vertices[1] = cy;
    vertices[2] = r;  vertices[3] = g; vertices[4] = b;

    for (int i = 1; i <= segments+1; i++) {
        float angle = 2.0f * M_PI * (i-1) / segments;
        float x = cx + rx * cosf(angle);
        float y = cy + ry * sinf(angle);
        int idx = i * 5;
        vertices[idx+0] = x;
        vertices[idx+1] = y;
        vertices[idx+2] = r;
        vertices[idx+3] = g;
        vertices[idx+4] = b;
    }
}

int generateSquares(float* vertices, int levels) {
    int count = 0;
    float step = 0.15f;
    for (int i = 0; i < levels; i++) {
        float s = 0.6f - i * step;
        float x0 = -s, y0 = -s;
        float x1 =  s, y1 =  s;

        vertices[count++] = x0; vertices[count++] = y0; vertices[count++] = 1; vertices[count++] = 1; vertices[count++] = 1;
        vertices[count++] = x1; vertices[count++] = y0; vertices[count++] = 1; vertices[count++] = 1; vertices[count++] = 1;
        vertices[count++] = x1; vertices[count++] = y1; vertices[count++] = 1; vertices[count++] = 1; vertices[count++] = 1;
        vertices[count++] = x0; vertices[count++] = y1; vertices[count++] = 1; vertices[count++] = 1; vertices[count++] = 1;
    }
    return count/5;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Shapes", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float triangle[] = {
        -0.2f, 0.6f,   1, 0, 0,
         0.2f, 0.6f,   0, 1, 0,
         0.0f, 0.9f,   0, 0, 1
    };

    unsigned int VAO1, VBO1;
    glGenVertexArrays(1, &VAO1);
    glGenBuffers(1, &VBO1);
    glBindVertexArray(VAO1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangle), triangle, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    int segments = 100;
    float ellipse[(102)*5];
    generateEllipse(ellipse, segments, -0.6f, 0.75f, 0.2f, 0.1f, 1, 0, 0);

    unsigned int VAO2, VBO2;
    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO2);
    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER,
VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ellipse), ellipse, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    float circle[(102)*5];
    generateEllipse(circle, segments, 0.6f, 0.75f, 0.15f, 0.15f, 1, 0, 0);

    unsigned int VAO3, VBO3;
    glGenVertexArrays(1, &VAO3);
    glGenBuffers(1, &VBO3);
    glBindVertexArray(VAO3);
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, sizeof(circle), circle, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    float squares[2000];
    int squareCount = generateSquares(squares, 5);

    unsigned int VAO4, VBO4;
    glGenVertexArrays(1, &VAO4);
    glGenBuffers(1, &VBO4);
    glBindVertexArray(VAO4);
    glBindBuffer(GL_ARRAY_BUFFER, VBO4);
    glBufferData(GL_ARRAY_BUFFER, sizeof(squares), squares, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    glLineWidth(25.0f);  
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glBindVertexArray(VAO1);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindVertexArray(VAO2);
        glDrawArrays(GL_TRIANGLE_FAN, 0, segments+2);

        glBindVertexArray(VAO3);
        glDrawArrays(GL_TRIANGLE_FAN, 0, segments+2);

        glBindVertexArray(VAO4);
        for (int i = 0; i < squareCount/4; i++) {
            glDrawArrays(GL_LINE_LOOP, i*4, 4);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}