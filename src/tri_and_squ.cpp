#include "config.h"

void framebuffer_size_callback(GLFWwindow *window, int width, int height);

const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";

const char *fragmentShaderSourceTriangle = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
    "}\0";  

const char *fragmentShaderSourceSquare = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);\n"
    "}\0";  

int main()
{
    float triangleVertices[] = {
        -0.8f, -0.8f, 0.0f,
         0.8f, -0.8f, 0.0f,
         0.0f,  1.0f, 0.0f  
    };

    float squareVertices[] = {
        0.4f,  0.4f, 0.0f,  
        0.8f,  0.4f, 0.0f,  
        0.8f,  0.8f, 0.0f,  
        0.4f,  0.8f, 0.0f   
    };

    unsigned int indices[] = {  
        0, 1, 2,  
        0, 2, 3   
    };

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    
    unsigned int fragmentShaderTriangle = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderTriangle, 1, &fragmentShaderSourceTriangle, NULL);
    glCompileShader(fragmentShaderTriangle);

    unsigned int fragmentShaderSquare = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderSquare, 1, &fragmentShaderSourceSquare, NULL);
    glCompileShader(fragmentShaderSquare);

    unsigned int shaderProgramTriangle = glCreateProgram();
    glAttachShader(shaderProgramTriangle, vertexShader);
    glAttachShader(shaderProgramTriangle, fragmentShaderTriangle);
    glLinkProgram(shaderProgramTriangle);

    unsigned int shaderProgramSquare = glCreateProgram();
    glAttachShader(shaderProgramSquare, vertexShader);
    glAttachShader(shaderProgramSquare, fragmentShaderSquare);
    glLinkProgram(shaderProgramSquare);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShaderTriangle);
    glDeleteShader(fragmentShaderSquare);

    unsigned int VAO1, VBO1;
    glGenVertexArrays(1, &VAO1);
    glGenBuffers(1, &VBO1);
    glBindVertexArray(VAO1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    unsigned int VAO2, VBO2, EBO2;
    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &EBO2);
    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.5f, 0.5f, 0.5f, 0.0f); 
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgramTriangle);
        glBindVertexArray(VAO1);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glUseProgram(shaderProgramSquare);
        glBindVertexArray(VAO2);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO1);
    glDeleteBuffers(1, &VBO1);
    glDeleteVertexArrays(1, &VAO2);
    glDeleteBuffers(1, &VBO2);
    glDeleteBuffers(1, &EBO2);
    glDeleteProgram(shaderProgramTriangle);
    glDeleteProgram(shaderProgramSquare);

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) { 
    glViewport(0, 0, width, height); 
}