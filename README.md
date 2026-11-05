# Leaf it to renewal
This the official repository for the paper [_Leaf it to renewal_](https://arxiv.org/abs/2509.20145).

I provide a tutorial notebook, which shows how to:
1. Create time-to-failure and prognostics data with the underlying simulator
2. Evaluate a policy's performance on this data for a single and multiple cost settings

I changed notation multiple times throughout the project, so the code notation does not fit with the paper in all cases. The most prominent case is $\hat{C}_\inf$ which denotes the final evaluated performance of each heuristic in the paper, is denoted as R in the code (following the notation from Kamariotis et al. (2024)).

Furthermore, I only included the heuristic code with the RUL-simulator integration. I skipped the small neural network case study on the C-MAPSS dataset, as I had to change the whole underlying code to make the interface work with the NN's output, which is now very messy. 

The most straightforward way to get the project up and running on your machine is to use the 'uv' manager. I provide the package requirements in the .toml and .lock files. 

I tried to document the my code extensively and give the used functions expressive names. However, if things are still not clear or if you have any troubles in running the code, then please contact me via daniel.koutas@tum.de.
