#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

// Window size
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// Subwindow size and position
const int SUB_X = 500;
const int SUB_Y = 100;
const int SUB_WIDTH = 250;
const int SUB_HEIGHT = 400;

// Ellipse vertices count
const int ELLIPSE_SEGMENTS = 100;

// Current subwindow background color
float subBgColor[3] = {0.2f, 0.3f, 0.4f};

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void drawEllipse(float cx, float cy, float rx, float ry)
{
    glBegin(GL_TRIANGLE_FAN);
    glColor3f(1.0f, 1.0f, 1.0f); // White ellipse
    glVertex2f(cx, cy);
    for (int i = 0; i <= ELLIPSE_SEGMENTS; ++i)
    {
        float theta = 2.0f * 3.1415926f * float(i) / float(ELLIPSE_SEGMENTS);
        float x = rx * cosf(theta);
        float y = ry * sinf(theta);
        glVertex2f(cx + x, cy + y);
    }
    glEnd();
}

// Handle key input to change subwindow background color
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        subBgColor[0] = 1.0f; subBgColor[1] = 0.0f; subBgColor[2] = 0.0f; // Red
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
    {
        subBgColor[0] = 0.0f; subBgColor[1] = 1.0f; subBgColor[2] = 0.0f; // Green
    }
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
    {
        subBgColor[0] = 0.0f; subBgColor[1] = 0.0f; subBgColor[2] = 1.0f; // Blue
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        subBgColor[0] = 1.0f; subBgColor[1] = 1.0f; subBgColor[2] = 1.0f; // White
    }
}

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // OpenGL 2 for immediate mode drawing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "GLFW Subwindow Example", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        // 1. Draw main window background
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glClearColor(0.6f, 0.6f, 0.6f, 1.0f); // Gray background
        glClear(GL_COLOR_BUFFER_BIT);

        // 2. Draw "subwindow" background (in viewport)
        glViewport(SUB_X, SUB_Y, SUB_WIDTH, SUB_HEIGHT);
        glClearColor(subBgColor[0], subBgColor[1], subBgColor[2], 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Set orthographic projection for easy 2D drawing in the subwindow area
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, SUB_WIDTH, 0, SUB_HEIGHT, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Draw white ellipse in center of subwindow
        drawEllipse(SUB_WIDTH / 2.0f, SUB_HEIGHT / 2.0f, SUB_WIDTH / 3.0f, SUB_HEIGHT / 4.0f);

        // Reset viewport back to full window for any further rendering if needed
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}