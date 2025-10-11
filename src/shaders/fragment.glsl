#version 330 core
out vec4 FragColor;
in vec3 ourColor;
in vec3 aColor;
in vec3 aPos;
in vec2 TexCoords;
uniform sampler2D ourTexture;
void main()
{
    FragColor = texture(ourTexture, TexCoords) * vec4(ourColor, 1.0);
};