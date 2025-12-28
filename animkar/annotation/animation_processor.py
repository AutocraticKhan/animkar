import random
import math

class AnimationProcessor:
    """Python version of the JavaScript AnimationProcessor for frame animation logic"""

    def __init__(self, frame_rate=30):
        self.FRAME_RATE = frame_rate

        # Movement timing constants
        self.TRANSITION_FRAMES = math.floor(0.5 * frame_rate)
        self.ANTICIPATION_FRAMES = math.floor(0.3 * frame_rate)

        # Zoom constants
        self.MAX_CONTENT_ZOOM = 1.15
        self.PUNCH_IN_ZOOM = 1.1
        self.ZOOM_IN_FRAMES = random.randint(15, 30)
        self.HOLD_FRAMES = math.floor(1.5 * frame_rate)
        self.ZOOM_OUT_FRAMES = self.ZOOM_IN_FRAMES * 2

        # Blink constants
        self.BLINK_DURATION_FRAMES = 5

    def initialize_columns(self, frames):
        """Initialize animation columns in frames"""
        for frame in frames:
            frame['head_direction'] = 'M'
            frame['eye_direction'] = 'M'
            frame['head_tilt'] = 0
            frame['zoom_level'] = 1.0
            frame['blink'] = False
        return frames

    def initialize_states(self):
        """Initialize animation state machines"""
        states = {}

        # Head position states for different modes
        states['bc'] = {  # big_center mode
            'position': 'M',
            'frames_in_position': 0,
            'time_since_last_shift': 0,
            'hold_duration': random.randint(3, 5),  # 3-5 seconds
            'shift_interval': random.randint(6, 8),  # 6-8 seconds
            'is_transitioning': False,
            'transition_frames_left': 0
        }

        states['bs'] = {  # big_side mode
            'position': 'M',
            'frames_in_position': 0,
            'hold_duration': random.randint(2, 4)  # 2-4 seconds
        }

        states['ss'] = {  # small_side mode
            'position': 'R',
            'frames_in_position': 0,
            'hold_duration': random.randint(10, 15)  # 10-15 seconds
        }

        # Eye movement states
        states['bc_eye'] = {
            'is_darting': False,
            'dart_frames_left': 0,
            'dart_direction': 'M',
            'time_since_last_dart': 0,
            'next_dart_time': random.randint(2, 3)  # 2-3 seconds
        }

        states['bs_eye'] = {
            'is_glancing_back': False,
            'glance_frames_left': 0,
            'time_since_last_glance': 0,
            'next_glance_time': random.randint(3, 5)  # 3-5 seconds
        }

        states['ss_eye'] = {
            'is_scanning': False,
            'scan_frames_left': 0,
            'scan_direction': 'R',
            'time_since_last_scan': 0,
            'next_scan_time': random.randint(2, 3)  # 2-3 seconds
        }

        # Head tilt state
        states['tilt'] = {
            'is_tilting': False,
            'tilt_frames_left': 0,
            'current_tilt_value': 0,
            'time_since_last_tilt': 0,
            'next_tilt_interval': random.randint(8, 10)  # 8-10 seconds
        }

        # Zoom state
        states['zoom'] = {
            'is_zooming': False,
            'zoom_frames_left': 0,
            'zoom_phase': None,
            'time_since_last_zoom': 1000
        }

        # Blink state
        states['blink'] = {
            'is_blinking': False,
            'blink_frames_left': 0,
            'time_since_last_blink': 0,
            'next_blink_interval': random.randint(3, 7),  # 3-7 seconds
            'is_double_blink_pending': False
        }

        return states

    def process_blink(self, frames, index, state):
        """Process blinking logic for a frame"""
        if state['is_blinking']:
            frames[index]['blink'] = True
            state['blink_frames_left'] -= 1
            if state['blink_frames_left'] <= 0:
                state['is_blinking'] = False
                state['time_since_last_blink'] = 0

                # Double blink handling
                if state['is_double_blink_pending']:
                    state['next_blink_interval'] = random.randint(7, 15)  # 7-15 frames for double blink
                    state['is_double_blink_pending'] = False
                else:
                    state['next_blink_interval'] = random.randint(3, 7)  # 3-7 seconds
            return

        state['time_since_last_blink'] += 1
        trigger_found = False

        # Trigger on head direction change (50% chance)
        if index > 0 and frames[index]['head_direction'] != frames[index - 1]['head_direction']:
            if random.random() < 0.5:
                trigger_found = True

        # Trigger on word start
        if not trigger_found and index > 0:
            if frames[index]['word'] and not frames[index - 1]['word']:
                trigger_found = True

        # Trigger on time interval
        if not trigger_found and state['time_since_last_blink'] >= state['next_blink_interval'] * self.FRAME_RATE:
            trigger_found = True

        if trigger_found:
            state['is_blinking'] = True
            state['blink_frames_left'] = self.BLINK_DURATION_FRAMES
            frames[index]['blink'] = True
            state['blink_frames_left'] -= 1

            # 15% chance of double blink
            if random.random() < 0.15:
                state['is_double_blink_pending'] = True

    def process_zoom_no_avatar(self, frames):
        """Apply smooth zoom to no_avatar blocks"""
        # Find no_avatar blocks
        blocks = []
        in_block = False
        block_start = 0

        for i, frame in enumerate(frames):
            if frame['mode'] == 'no_avatar' and not in_block:
                in_block = True
                block_start = i
            elif frame['mode'] != 'no_avatar' and in_block:
                in_block = False
                blocks.append([block_start, i - 1])

        if in_block:
            blocks.append([block_start, len(frames) - 1])

        # Apply smooth zoom to each block
        for start, end in blocks:
            block_length = end - start + 1
            if block_length < 2:
                continue

            midpoint = block_length // 2

            # Zoom in
            for i in range(midpoint):
                progress = i / (midpoint - 1) if midpoint > 1 else 0
                frames[start + i]['zoom_level'] = 1.0 + (self.MAX_CONTENT_ZOOM - 1.0) * progress

            # Zoom out
            for i in range(block_length - midpoint):
                progress = i / (block_length - midpoint - 1) if block_length - midpoint > 1 else 0
                frames[start + midpoint + i]['zoom_level'] = self.MAX_CONTENT_ZOOM - (self.MAX_CONTENT_ZOOM - 1.0) * progress

    def apply_eye_anticipation(self, frames):
        """Apply eye anticipation for big_side mode turns"""
        # Find turn indices for big_side mode
        turn_indices = []
        for i, frame in enumerate(frames):
            if (frame['mode'] == 'big_side' and
                frame['head_direction'] == 'R' and
                i > 0 and frames[i - 1]['head_direction'] == 'M'):
                turn_indices.append(i)

        # Apply anticipation
        for index in turn_indices:
            start_index = max(0, index - self.ANTICIPATION_FRAMES)
            for i in range(start_index, index):
                frames[i]['eye_direction'] = 'R'

    def process(self, frames):
        """Main processing function - apply all animations"""
        # Initialize columns
        frames = self.initialize_columns(frames)

        # Initialize state machines
        states = self.initialize_states()

        # Pre-process zoom for no_avatar blocks
        self.process_zoom_no_avatar(frames)

        # Main frame-by-frame loop
        for index, frame in enumerate(frames):
            mode = frame['mode']

            # Set default eye direction to follow head
            frames[index]['eye_direction'] = frames[index]['head_direction']

            # === BLINK ===
            self.process_blink(frames, index, states['blink'])

        # Post-processing: Eye anticipation
        self.apply_eye_anticipation(frames)

        return frames
