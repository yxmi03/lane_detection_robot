import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor():
    def __init__(self, PWM_A, DIR_A, PWM_B, DIR_B):
        """
        Initialize motor driver with Cytron 10A pins
        PWM_A, PWM_B: PWM pins for speed control
        DIR_A, DIR_B: Direction pins for motor direction
        """
        self.PWM_A = PWM_A
        self.DIR_A = DIR_A
        self.PWM_B = PWM_B
        self.DIR_B = DIR_B
        
        # Setup GPIO pins
        GPIO.setup(self.PWM_A, GPIO.OUT)
        GPIO.setup(self.DIR_A, GPIO.OUT)
        GPIO.setup(self.PWM_B, GPIO.OUT)
        GPIO.setup(self.DIR_B, GPIO.OUT)
        
        # Setup PWM
        self.pwmA = GPIO.PWM(self.PWM_A, 100)  # 100Hz frequency
        self.pwmB = GPIO.PWM(self.PWM_B, 100)
        
        # Start PWM with 0% duty cycle
        self.pwmA.start(0)
        self.pwmB.start(0)
        self.mySpeed = 0

    def move(self, speed=0.5, turn=0, t=0):
        """
        Move the robot with given speed and turn values
        speed: -1 to 1 (converted to -100 to 100)
        turn: -1 to 1 (converted to -70 to 70)
        t: time to move in seconds
        """
        speed *= 100
        turn *= 70
        leftSpeed = speed - turn
        rightSpeed = speed + turn
        
        # Constrain speeds to -100 to 100
        leftSpeed = min(max(leftSpeed, -100), 100)
        rightSpeed = min(max(rightSpeed, -100), 100)
        
        # Set motor directions using DIR pins
        GPIO.output(self.DIR_A, leftSpeed > 0)
        GPIO.output(self.DIR_B, rightSpeed > 0)
        
        # Set motor speeds (always positive for PWM)
        self.pwmA.ChangeDutyCycle(abs(leftSpeed))
        self.pwmB.ChangeDutyCycle(abs(rightSpeed))
        
        sleep(t)

    def stop(self, t=0):
        """Stop both motors"""
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        self.mySpeed = 0
        sleep(t)

def main():
    """Test function for the motor driver"""
    motor.move(0.5, 0, 2)    # Forward
    motor.stop(2)
    motor.move(-0.5, 0, 2)   # Backward
    motor.stop(2)
    motor.move(0, 0.5, 2)    # Turn right
    motor.stop(2)
    motor.move(0, -0.5, 2)   # Turn left
    motor.stop(2)

if __name__ == '__main__':
    # Example pin configuration - adjust these to match your wiring
    motor = Motor(2, 3, 17, 22)  # PWM_A, DIR_A, PWM_B, DIR_B
    main()
