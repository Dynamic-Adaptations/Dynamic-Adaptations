1. # Dynamic Reading Mode Userscript

   An intelligent reading assistant plugin based on face distance detection that automatically adjusts font size and contrast according to the user's distance from the screen, providing an optimal reading experience.

   ## Features

   1. **Reading Mode**: Extracts main content from web pages, removes distracting elements, and provides a clean reading interface
   2. **Dynamic Font Adjustment**: Automatically adjusts font size based on user's distance from the screen
   3. **Dynamic Contrast Adjustment**: Automatically adjusts contrast based on user's distance from the screen
   4. **Reading Timer**: Records reading time
   5. **Reading History**: Saves reading history and statistical data

   ## Installation

   ### 1. Install Tampermonkey Browser Extension

   First, you need to install the Tampermonkey browser extension:

   - [Tampermonkey for Chrome](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo)
   - [Tampermonkey for Firefox](https://addons.mozilla.org/en-US/firefox/addon/tampermonkey/)
   - [Tampermonkey for Edge](https://microsoftedge.microsoft.com/addons/detail/tampermonkey/iikmkjmpaadaobahmlepeloendndfphd)

   ### 2. Install Script

   #### Method 1: Direct Installation

   1. Ensure Tampermonkey extension is installed
   2. Open the `dynamic-reading-mode.user.js` file
   3. Tampermonkey will automatically show the installation interface, click the "Install" button

   #### Method 2: Manual Installation

   1. Open Tampermonkey extension control panel
   2. Click the "Create a new script" tab
   3. Delete the default code in the editor
   4. Paste the complete code from `dynamic-reading-mode.user.js` into the editor
   5. Press `Ctrl+S` (or `Command+S`) to save the script

   ## Usage Instructions

   After installation, a blue floating button will appear on any webpage (bottom right corner). Click the button to expand the menu:

   ### Reading Mode

   Click the "Enter Reading Mode" button to convert the page to reading mode, removing distracting elements for a better reading experience.

   ### Dynamic Font Adjustment

   1. Calibration is required before first use
   2. Click the "Calibrate Distance" button
   3. Follow the prompts to calibrate
   4. After calibration is complete, check the "Dynamic Font Adjustment" option to enable the feature

   ### Dynamic Contrast Adjustment

   Enable the "Dynamic Contrast Adjustment" option, and the system will automatically adjust contrast based on distance.

   ## Important Notes

   - Camera permission authorization required
   - Requires a modern browser that supports MediaPipe (such as Chrome, Edge, Firefox, etc.)
   - Dynamic adjustment features require a front-facing camera

   ## Privacy Statement

   - All facial detection and distance calculations are performed locally and no data is sent to remote servers
   - Calibration data and reading history are stored in the browser's local storage
   - This plugin does not collect any personal information

   ## Troubleshooting

   If you encounter issues, please try the following steps:

   1. Ensure camera permission is granted to the browser
   2. Ensure your device has a front-facing camera
   3. Try refreshing the page
   4. Recalibrate the distance
   5. If the problem persists, try reinstalling the plugin
