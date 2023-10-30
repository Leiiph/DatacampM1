# DatacampM1
Github repository for the Datacamp M1 project on music recommendation

# Goal of the project
The goals of our project are the following:
- With the user's inputs (artist name, song name and/or lyrics), provide music recommendation to them based on the mood of the song
- Be easy to use, so that everyone may use it without issues
- Be fast
- Respects the environment

# How we achieve this
We use two main things: Deep Learning and another dataset we found on kaggle here: https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019/data
This dataset is used when the user enters a song name and an artist name. If we have them in our dataset, we will collect the lyrics in it. Else, we will be using the lyrics given by the user.
After that, and only if we don't have the song in the dataset, we use our model to get the lyrics' mood and then find in our dataset matching songs based oon that mood.

# How to use Vibe
When the website is launched, you just have to enter the song name and artist name in the fields provided for those two actions. If you want to, you may add lyrics by clicking on "add lyrics (optional)", but if you are sure you song is in our dataset, you can just press enter to launch the computations.

# Accuracy of the model
Currently, our model has an accuracy of 60%!
