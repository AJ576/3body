import pygame
import numpy as np
import time

# Constants
WIDTH, HEIGHT = 800, 750
FPS = 60
G = 6.6743e-11  # Gravitational constant
SCALE = 1e9  # Scale factor for positions
TIMESTEP = 1000  # Initial time step in seconds
SLIDER_WIDTH = 300
SLIDER_HEIGHT = 10
PROXIMA_MIN_MASS = 2.38e29  # Minimum mass of a star (~0.12 solar masses)
MASS_RADIUS_SCALE = 1e-29  # Scale factor for radius
PLANET_MASS_RADIUS_SCALE = 5e-28  # Scale factor for radius
MASS_COLOR_SCALE = 1e30  # Scale factor for color transitions

PLUTO_MIN_MASS = 1.31e22  # Minimum mass of a planet (Pluto)
JUPITER_MAX_MASS = 1.898e27 # Maximum mass of a planet (Jupiter)

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
font = pygame.font.SysFont("Arial", 16)

# Slider setup
slider_rect = pygame.Rect(WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - 50, SLIDER_WIDTH, SLIDER_HEIGHT)
slider_thumb_rect = pygame.Rect(WIDTH // 2 - SLIDER_WIDTH // 2 + TIMESTEP // 1000, HEIGHT - 55, 10, 20)
dragging_slider = False  # Slider dragging state

# Add object mode: "star" or "planet"
current_mode = "star"

def mass_to_color(mass):
    if mass < 1 * MASS_COLOR_SCALE:
        # Red → Orange
        return (255, int(165 * (mass / MASS_COLOR_SCALE)), 0)  
    elif mass < 2 * MASS_COLOR_SCALE:
        # Orange → Yellow
        return (255, 165 + int(90 * ((mass - MASS_COLOR_SCALE) / MASS_COLOR_SCALE)), 0)
    elif mass < 3 * MASS_COLOR_SCALE:
        # Yellow → White
        green = 255
        blue = int(255 * ((mass - 2 * MASS_COLOR_SCALE) / MASS_COLOR_SCALE))
        return (255, green, blue)
    elif mass < 4 * MASS_COLOR_SCALE:
        # White → Blue
        red = 255 - int(255 * ((mass - 3 * MASS_COLOR_SCALE) / MASS_COLOR_SCALE))
        return (red, 255, 255)
    else:
        # Max mass: Pure Blue
        return (0, 0, 255)



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
    base_radius = max(1, int(mass * MASS_RADIUS_SCALE))  # Radius based on mass
    scaled_radius = base_radius / (SCALE / 1e9)  # Adjust radius based on zoom level
    return max(2, int(scaled_radius))  # Ensure radius doesn't go to 0

def calculate_dynamic_radius_planet(mass):
    """Calculate the radius dynamically for planets."""
    base_radius = mass * PLANET_MASS_RADIUS_SCALE # Radius based on mass
    scaled_radius = base_radius / (SCALE / 1e9)  # Adjust radius based on zoom level
    return max(2, int(scaled_radius))  # Ensure radius doesn't go to 0


def draw_bodies(screen, bodies, preview_data=None):
    screen.fill((0, 0, 0))  # Black background
    for body in bodies:
        x, y = (body["state"][:2] - camera_offset) / SCALE + np.array([WIDTH // 2, HEIGHT // 2])
        x, y = int(x), int(y)

        radius = calculate_dynamic_radius(body["mass"])
        # Ensure the object is drawn until the entire circle is out of the screen borders
        if -radius <= x < WIDTH + radius and -radius <= y < HEIGHT + radius:
            if body["type"] == "planet":
                radius = calculate_dynamic_radius_planet(body["mass"])
            else:   
                radius = calculate_dynamic_radius(body["mass"])
            pygame.draw.circle(screen, body["color"], (x, y), radius)
    
    # Draw preview (if any)
    if preview_data:
        pos, velocity, mass = preview_data
        if current_mode == "planet":
            radius = calculate_dynamic_radius_planet(mass)
        else:   
            radius = calculate_dynamic_radius(mass)
        color = mass_to_color(mass)
        pygame.draw.circle(screen, color, pos, radius)
        drag_end = (pos[0] + velocity[0] / 1e3, pos[1] + velocity[1] / 1e3)
        pygame.draw.line(screen, (255, 255, 255), pos, drag_end, 2)
    
    # Draw the slider
    pygame.draw.rect(screen, (255, 255, 255), slider_rect)  # Slider background
    pygame.draw.rect(screen, (0, 255, 0), slider_thumb_rect)  # Slider thumb


    if focused_body_index != -1:
        focused_body = bodies[focused_body_index]
        mass_text = font.render(f"Mass: {focused_body['mass']:.2e} kg", True, (255, 255, 255))
        velocity_text = font.render(f"Velocity: ({focused_body['state'][2]:.2f}, {focused_body['state'][3]:.2f}) m/s", True, (255, 255, 255))
        type_text = font.render(f"Type: {'Star' if focused_body['mass'] >= PROXIMA_MIN_MASS else 'Planet'}", True, (255, 255, 255))
        
        screen.blit(mass_text, (WIDTH - 300, 10))
        screen.blit(velocity_text, (WIDTH - 300, 30))
        screen.blit(type_text, (WIDTH - 300, 50))
    if preview_data:
        mass_text = font.render(f"Body in creation: Mass: {preview_data[2]:.2e} kg", True, (255, 255, 255))
        screen.blit(mass_text, (10, 10))
        if focused_body_index == -1:
            mass_text = font.render("Body in creation: Mass: ", True, (255, 255, 255))
            screen.blit(mass_text, (10, 10))
    scale = font.render(f"Scale: {SCALE:.2e} m", True, (255, 255, 255))
    screen.blit(scale, (10, HEIGHT - 20))
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
            "radius": calculate_dynamic_radius(mass),  # Dynamic radius
            "type":"star"
        })
    elif mode == "planet":
        mass = min(JUPITER_MAX_MASS, mass)
        bodies.append({
            "mass": mass,
            "state": np.array([(position[0] - WIDTH // 2) * SCALE + camera_offset[0], 
                               (position[1] - HEIGHT // 2) * SCALE + camera_offset[1], 
                               velocity[0], velocity[1]]),  # Position and velocity
            "color": (200, 200, 0),  # Yellowish
            "radius": calculate_dynamic_radius_planet(mass),  # Dynamic radius
            "type":"planet"
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
                if bodies:  # Ensure there are bodies to focus on
                    start_index = focused_body_index  # Remember the starting point
                    while True:
                        print(f"focus:{focused_body_index}")
                        print(f"start{start_index}")
                        focused_body_index = (focused_body_index + 1) % len(bodies)
                        # Check if the current body matches the desired type
                        if bodies[focused_body_index]["type"] == current_mode:
                            break
                        # If we've looped through all bodies and found none, stop
                        if focused_body_index == start_index:
                            focused_body_index = -1  # No matching body found
                            print("No matching body found")
                            break
                        if start_index == -1:
                            start_index = 0
            elif event.key == pygame.K_u:  # Unfocus the camera
                focused_body_index = -1
            elif event.key == pygame.K_x:  # Delete the focused body
                if focused_body_index != -1:
                    del bodies[focused_body_index]
                    focused_body_index = -1  # Unfocus after deletion
                    
            elif event.key == pygame.K_EQUALS:  # Zoom in
                SCALE = max(SCALE / 1.1, 1e3)  # Clamp to a minimum scale
            elif event.key == pygame.K_MINUS:  # Zoom out
                SCALE = min(SCALE * 1.1, 1e11)  # Clamp to a maximum scale


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
                    raw_velocity = ((release_position[0] - click_position[0]) * 1e3, 
                                    (release_position[1] - click_position[1]) * 1e3)

                    # Adjust for relative velocity if a body is focused
                    if focused_body_index != -1:
                        focused_body_velocity = bodies[focused_body_index]["state"][2:4]
                        velocity = (raw_velocity[0] + focused_body_velocity[0], 
                                    raw_velocity[1] + focused_body_velocity[1])  # Add focus velocity
                    else:
                        velocity = raw_velocity

                    # Add the new body
                    add_body(click_position, velocity, current_mode, preview_mass)
                    click_position = None
        elif event.type == pygame.MOUSEMOTION:
            if dragging_slider:
                new_x = max(min(event.pos[0], slider_rect.right - slider_thumb_rect.width), slider_rect.left)
                slider_thumb_rect.x = new_x
                TIMESTEP = (slider_thumb_rect.x - slider_rect.left) * 100  # Adjust time step
            elif click_position:
                hold_duration = time.time() - mouse_down_time
                if current_mode == "star":
                    preview_mass = PROXIMA_MIN_MASS + hold_duration * 3e29  # Increment mass with time held
                else:
                    max_hold_time = 15  # Time in seconds to reach max mass
                    mass_range = JUPITER_MAX_MASS - PLUTO_MIN_MASS

                    # Use a quadratic scaling for smoother and slower initial growth
                    progress = min(hold_duration / max_hold_time, 1)  # Progress normalized between 0 and 1
                    scaled_increment = mass_range * (progress ** 4)  # Quadratic growth for gradual scaling
                    
                    # Update the preview mass, capping it at Jupiter's max mass
                    preview_mass = min(PLUTO_MIN_MASS + scaled_increment, JUPITER_MAX_MASS)
                drag_position = event.pos
                raw_velocity = ((drag_position[0] - click_position[0]) * 1e3, 
                                (drag_position[1] - click_position[1]) * 1e3)

                # Relative velocity calculation
                if focused_body_index != -1:
                    focused_body_velocity = bodies[focused_body_index]["state"][2:4]
                    preview_velocity = (raw_velocity[0], raw_velocity[1])  # Start preview line at 0
                else:
                    preview_velocity = raw_velocity

                # Prepare preview data
                preview_data = (click_position, preview_velocity, preview_mass)


                
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
    