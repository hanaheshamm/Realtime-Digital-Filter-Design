# Realtime Digital Filter Design

## Overview

The Realtime Digital Filter Design application is a sophisticated tool aimed at signal processing professionals, educators, and enthusiasts. This desktop application facilitates the design of custom digital filters through intuitive zeros and poles placement on the z-plane. It combines user-friendly interaction with powerful visualization to streamline the filter design process, offering real-time insights into the filter's frequency response and its effect on signals.

## Key Features

1. **Z-Plane Visualization and Interaction**
    - **Interactive Plot:** Users can interactively place zeros and poles on the z-plane, equipped with a unit circle for reference. This direct manipulation allows for an intuitive design process, bridging the gap between theory and practical application.
    - **Modifications:** Features include the ability to drag and modify the placement of zeros/poles, delete individual elements, and clear selections with options for zeros, poles, or both. Additionally, users can opt to automatically add conjugates for complex elements, enhancing the filter's symmetry and response characteristics.

2. **Frequency Response Visualization**
    - **Dual Graphs:** The application provides two separate graphs displaying the magnitude and phase response of the designed filter. This instant feedback on the filter's frequency response is crucial for understanding its behavior and ensuring it meets the desired specifications.

3. **Real-Time Filtering Simulation**
    - **Signal Processing:** Users can apply the designed filter to a lengthy signal (minimum of 10,000 points), simulating a real-time filtering process. This includes a graphical representation of the signal's time progress and the filtered signal's response, offering a dynamic way to observe the filter's effects over time.
    - **Controlled Speed:** The filtering speed/temporal resolution is adjustable via a slider, allowing users to observe the filtering process at various rates, from 1 point per second to 100 points per second or any rate in between.
    - **Input Signal Generation:** An innovative feature where users can generate arbitrary real-time signals through mouse movement over a small padding area. The generated signal's frequency content varies with the speed of the mouse movement, providing a hands-on way to test the filter's performance across different frequencies.

4. **Phase Correction with All-Pass Filters**
    - **All-Pass Filter Library:** The application includes a library of all-pass filters, enabling users to visualize and select from pre-designed options or add them to the filter design to correct phase discrepancies.
    - **Custom All-Pass Design:** Users not satisfied with the library options can design their own all-pass filter by providing an arbitrary coefficient "a." The application calculates its phase response and integrates it with the existing filter design.
    - **Toggle Integration:** Added all-pass elements can be enabled or disabled through a drop-menu or checkboxes group, offering flexibility in fine-tuning the filter's overall phase response.

##  Key Feature Insights

- **Z-Plane Interaction:** Provides a fundamental understanding of filter design by allowing users to experiment with zeros and poles placement, directly observing the impact on the filter's characteristics.
- **Frequency Response Visualization:** Essential for validating the filter's design, ensuring that it meets the desired specifications in terms of magnitude and phase response.
- **Real-Time Filtering Simulation:** Offers a practical perspective by demonstrating the filter's performance on actual signals, crucial for applications requiring dynamic signal processing.
- **Phase Correction with All-Pass Filters:** Addresses the common challenge of phase distortion in filter design, enabling users to achieve a more precise control over the filter's phase response without affecting its magnitude response.



## Examples and Inspiration

This project draws inspiration from existing digital filter design tools such as those available at EarLevel Engineering. These resources provide valuable insights into user interaction and visualization techniques, informing the development of a user-friendly and functional filter design application.

## Usage
- **Designing Filters:** Begin by plotting zeros and poles on the z-plane to design your filter.
- **Adjusting Filter Properties:** Utilize the interactive features to fine-tune your filter's design.
- **Applying the Filter:** Test your filter on a signal of your choice, observing the filtering process in real time.
- **Incorporating All-Pass Filters:** Enhance your filter design with phase correction through the application's all-pass filter library or by designing your own.


## Conclusion

The Realtime Digital Filter Design application represents a significant advancement in digital filter design software, offering a comprehensive suite of features that empower users to create, visualize, and test digital filters in a dynamic and interactive environment.

## Installation

```bash
# Clone the project repository
git clone [repository-link]
# Change directory to the project folder
cd RealtimeDigitalFilterDesign
# Install the required dependencies
pip install -r requirements.txt
# Execute the application
python main.py

