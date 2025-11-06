#!/usr/bin/env python3
"""
Simple Dash Launcher Script
Run this after your Flask app is running to launch the interactive timetable viewer
"""

import sys
import os

def main():
    """Launch the Dash interface"""
    print("🎓 Launching Interactive Timetable Viewer")
    print("="*50)
    
    # Add current directory to Python path
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    try:
        # Import the Flask app to access the helper functions
        print("📡 Importing Flask app...")
        import PAU_Timetable_Scheduler.app as app
        
        # Check if Flask app has the launch function
        if not hasattr(app, 'console_launch_dash'):
            print("❌ Flask app doesn't have dash launch functions")
            print("💡 Make sure you added the helper code to app.py")
            return 1
        
        # Launch Dash interface with latest upload
        print("🚀 Starting Dash interface...")
        success = app.console_launch_dash()
        
        if success:
            print("✅ Dash interface finished")
            return 0
        else:
            print("❌ Failed to launch Dash interface")
            return 1
            
    except ImportError as e:
        print(f"❌ Could not import Flask app: {e}")
        print("💡 Make sure:")
        print("   1. You're in the correct directory")
        print("   2. Your Flask app is named 'app.py'")
        print("   3. You added the helper functions to app.py")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Launcher stopped by user")
        sys.exit(0)