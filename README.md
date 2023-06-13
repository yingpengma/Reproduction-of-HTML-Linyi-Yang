# HTML_partial_reproduction

This is an **unofficial** implementation for the paper [**HTML: Hierarchical Transformer-based Multi-task Learning for Volatility Prediction**](https://dl.acm.org/doi/abs/10.1145/3366423.3380128) (Yang et al., WWW 2020). Please refer to the **official** implementation [**HERE**](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction).

Previously, during my reproduction of HTML, I encountered some version issues that caused problems, which I fixed and successfully ran HTML.

However, as my previous code underwent significant changes for my own tasks, it may not be very useful for someone who simply wants to reproduce the original HTML. 
Therefore, I attempted to reproduce HTML as much as possible based on the [**original HTML repository**](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction), and was able to fully run one task (3 days stock **movement*** prediction), which is the code you are currently viewing.

I believe this code could provide enough guidance for you to fully reproduce the entire original HTML. By comparing the differences between the two repositories, you can definitely see how I solved some of the problems in the original repository.

***Note**: 
The label in the task here is **NOT** the **volatility** used in HTML, but the **movement** used in my own task, which is simply the label for _**whether the stock price rises or falls**_. 
If you want to reproduce the original HTML, you can go to the `tools.py` file I provided and make simple modifications to the `calculate_movement` function. 
Trust me, it's easy as long as you understand the difference between volatility and movement :)

PS: Due to the large size of the files, `sorted_list_3days.npy` and `sorted_embed_3days.npy` cannot be uploaded. You can generate these files yourself. The file `EarningsCallData\ReleasedDataset\ReleasedDataset_mp3` needs to be downloaded from https://github.com/GeminiLn/EarningsCall_Dataset.

Hope this repository can be helpful to you, good luck!
