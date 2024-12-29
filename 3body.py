import pygame
import numpy as np
import time

# Constants
WIDTH, HEIGHT = 800, 800
FPS = 60
G = 6.6743e-11  # Gravitational constant
SCALE = 1e9  # Scale factor for positions
TIMESTEP = 1000  # Initial time step in seconds
SLIDER_WIDTH = 300
SLIDER_HEIGHT = 10
PROXIMA_MIN_MASS = 2.38e29  # Minimum mass of a star (~0.12 solar masses)
MASS_RADIUS_SCALE = 1e-29  # Scale factor for radius
MASS_COLOR_SCALE = 1e30  # Scale factor for color transitions

# Initial bodies
bodies = []
simulation_running = False
camera_offset = np.array([0, 0])  # Offset for camera movement
focused_body_index = -1  # Index of the currently focused body (-1 means no focus)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Three-Body Simulation")
clock = pygame.time.Clock()

# Slider setup
slider_rect = pygame.Rect(WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - 50, SLIDER_WIDTH, SLIDER_HEIGHT)
slider_thumb_rect = pygame.Rect(WIDTH // 2 - SLIDER_WIDTH // 2 + TIMESTEP // 1000, HEIGHT - 55, 10, 20)
dragging_slider = False  # Slider dragging state

# Add object mode: "star" or "planet"
current_mode = "star"

def mass_to_color(mass):
    if mass < 1 * MASS_COLOR_SCALE:
        return (255, int(255 * (mass / MASS_COLOR_SCALE)), 0)  # Red → Orange
    elif mass < 2 * MASS_COLOR_SCALE:
        return (255 - int(255 * ((mass - MASS_COLOR_SCALE) / MASS_COLOR_SCALE)), 255, 0)  # Orange → Yellow
    elif mass < 3 * MASS_COLOR_SCALE:
        return (0, 255, int(255 * ((mass - 2 * MASS_COLOR_SCALE) / MASS_COLOR_SCALE)))  # Yellow → Green
    elif mass < 4 * MASS_COLOR_SCALE:
        return (0, 255 - int(255 * ((mass - 3 * MASS_COLOR_SCALE) / MASS_COLOR_SCALE)), 255)  # Green → Cyan
    else:
        return (0, 0, 255)  # Blue (Max Mass)


def compute_forces(bodies):
    forces = []
    for i, body1 in enumerate(bodies):
        force = np.zeros(2)
        for j, body2 in enumerate(bodies):
            if i != j:
                r_vec = body2["state"][:2] - body1["state"][:2]
                r_mag = np.linalg.norm(r_vec)
                if r_mag < 1e7:  # Softening factor to avoid singularities
                    r_mag = 1e7
                force += G * body1["mass"] * body2["mass"] * r_vec / r_mag**3
        forces.append(force)
    return np.array(forces)

def update_bodies(bodies, forces):
    for i, body in enumerate(bodies):
        acceleration = forces[i] / body["mass"]
        body["state"][2:] += acceleration * TIMESTEP  # Update velocity
        body["state"][:2] += body["state"][2:] * TIMESTEP  # Update position

def calculate_dynamic_radius(mass):
    """Calculate the radius dynamically based on mass and the current scale."""
    base_radius = max(5, int(mass * MASS_RADIUS_SCALE))
    scaled_radius = int(base_radius * (1e9 / SCALE))
    return max(1, scaled_radius)  # Ensure radius doesn't go to 0

def draw_bodies(screen, bodies, preview_data=None):
    screen.fill((0, 0, 0))  # Black background
    for body in bodies:
        x, y = (body["state"][:2] - camera_offset) / SCALE + np.array([WIDTH // 2, HEIGHT // 2])
        x, y = int(x), int(y)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            radius = calculate_dynamic_radius(body["mass"])
            pygame.draw.circle(screen, body["color"], (x, y), radius)
    
    # Draw preview (if any)
    if preview_data:
        pos, velocity, mass = preview_data
        radius = calculate_dynamic_radius(mass)
        color = mass_to_color(mass)
        pygame.draw.circle(screen, color, pos, radius)
        drag_end = (pos[0] + velocity[0] / 1e3, pos[1] + velocity[1] / 1e3)
        pygame.draw.line(screen, (255, 255, 255), pos, drag_end, 2)
    
    # Draw the slider
    pygame.draw.rect(screen, (255, 255, 255), slider_rect)  # Slider background
    pygame.draw.rect(screen, (0, 255, 0), slider_thumb_rect)  # Slider thumb
    pygame.display.flip()

def add_body(position, velocity, mode, mass=None):
    global focused_body_index
    if mode == "star":
        mass = mass if mass else PROXIMA_MIN_MASS
        bodies.append({
            "mass": mass,
            "state": np.array([(position[0] - WIDTH // 2) * SCALE + camera_offset[0], 
                               (position[1] - HEIGHT // 2) * SCALE + camera_offset[1], 
                               velocity[0], velocity[1]]),  # Position and velocity
            "color": mass_to_color(mass),
            "radius": calculate_dynamic_radius(mass)  # Dynamic radius
        })
    elif mode == "planet":
        mass = 5.972e24  # Earth-like planet mass
        bodies.append({
            "mass": mass,
            "state": np.array([(position[0] - WIDTH // 2) * SCALE + camera_offset[0], 
                               (position[1] - HEIGHT // 2) * SCALE + camera_offset[1], 
                               velocity[0], velocity[1]]),  # Position and velocity
            "color": (200, 200, 0),  # Yellowish
            "radius": calculate_dynamic_radius(mass)  # Dynamic radius
        })

    focused_body_index = len(bodies) - 1  # Focus on the last added body



# Main simulation loop
running = True
click_position = None
mouse_down_time = 0  # Track mouse hold duration for mass scaling
preview_mass = 0  # For live mass preview
preview_velocity = (0, 0)  # For velocity preview
while running:
    preview_data = None  # Reset preview data each frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:  # Move camera up
                if focused_body_index == -1:
                    camera_offset[1] -= 50 * SCALE
            elif event.key == pygame.K_s:  # Move camera down
                if focused_body_index == -1:
                    camera_offset[1] += 50 * SCALE
            elif event.key == pygame.K_a:  # Move camera left
                if focused_body_index == -1:
                    camera_offset[0] -= 50 * SCALE
            elif event.key == pygame.K_d:  # Move camera right
                if focused_body_index == -1:
                    camera_offset[0] += 50 * SCALE
            elif event.key == pygame.K_SPACE:  # Toggle simulation running
                simulation_running = not simulation_running
            elif event.key == pygame.K_p:  # Switch to planet mode
                current_mode = "planet"
            elif event.key == pygame.K_c:  # Switch to star mode
                current_mode = "star"
            elif event.key == pygame.K_TAB:  # Toggle focus between bodies
                focused_body_index = (focused_body_index + 1) % len(bodies) if bodies else -1
            elif event.key == pygame.K_u:  # Unfocus the camera
                focused_body_index = -1
            elif event.key == pygame.K_x:  # Delete the focused body
                if focused_body_index != -1:
                    del bodies[focused_body_index]
                    focused_body_index = -1  # Unfocus after deletion
            elif event.key == pygame.K_EQUALS:  # Zoom in
                SCALE = max(SCALE / 1.1, 1e7)  # Clamp to a minimum scale
            elif event.key == pygame.K_MINUS:  # Zoom out
                SCALE = min(SCALE * 1.1, 1e12)  # Clamp to a maximum scale


        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left-click
                if slider_thumb_rect.collidepoint(event.pos):
                    dragging_slider = True
                else:
                    click_position = event.pos
                    mouse_down_time = time.time()  # Start tracking hold duration
                    preview_mass = PROXIMA_MIN_MASS  # Reset preview mass
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left-click release
                if dragging_slider:
                    dragging_slider = False
                elif click_position:
                    # Calculate velocity based on drag
                    release_position = event.pos
                    velocity = ((release_position[0] - click_position[0]) * 1e3, 
                                (release_position[1] - click_position[1]) * 1e3)
                    add_body(click_position, velocity, current_mode, preview_mass)
                    click_position = None
        elif event.type == pygame.MOUSEMOTION:
            if dragging_slider:
                new_x = max(min(event.pos[0], slider_rect.right - slider_thumb_rect.width), slider_rect.left)
                slider_thumb_rect.x = new_x
                TIMESTEP = (slider_thumb_rect.x - slider_rect.left) * 100  # Adjust time step
            elif click_position:
                hold_duration = time.time() - mouse_down_time
                preview_mass = PROXIMA_MIN_MASS + hold_duration * 1e30  # Increment mass with time held
                drag_position = event.pos
                preview_velocity = ((drag_position[0] - click_position[0]) * 1e3, 
                                     (drag_position[1] - click_position[1]) * 1e3)
                preview_data = (click_position, preview_velocity, preview_mass)  # Prepare preview data

    if simulation_running and len(bodies) > 0:
        forces = compute_forces(bodies)  # Compute gravitational forces
        update_bodies(bodies, forces)  # Update positions and velocities
    
    # Update camera to follow focused body
    if focused_body_index != -1:
        camera_offset = bodies[focused_body_index]["state"][:2].copy()

    # Render everything
    draw_bodies(screen, bodies, preview_data)
    clock.tick(FPS)

pygame.quit()
    