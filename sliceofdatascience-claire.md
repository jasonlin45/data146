# Summary
In this slice of data science, Claire went over an introduction to deep learning using Tensorflow Keras, then an introduction to convolutional neural networks, transfer learning,
and ensembling. She started off with an example of a project she had done for MCM, a math modeling competition, involving Spotify data.  From this example, it was simple to see how each feature 
would contribute to the overall model.  Afterwards, she showed how this same technique might be applied to images.  Since images can be represented as 2d matrices, there needs to
be a way to map these matrices to features where relative location is preserved.  The solution comes in the convolutional filter, which can also be applied to multiple bands of 
each image as well.  They help create the features we need for our model.  Keras wraps all the math and complex implementation neatly under the hood, so it becomes very easy to 
create models.  

Additionally, Claire went over a technique called transfer learning.  Basically, a predefined model architecture is pretrained on a dataset (commonly imagenet),
and these models can be used to build our own models.  Within this, there are three different approaches.  One allows the entire model to be trainable (will likely require a larger
dataset), the second opts for fine-tuning where only a portion of the prebuilt model is frozen, using those layers as feature extractors.  The final approach freezes the entire
prebuild model, using the entire model as a feature extractor.  In all of these cases, additional custom classifier layers can be built on to adapt the classification for our use case.

Finally, she covered ensembling multiple models together to get a higher accuracy.  By taking into account the estimates of multiple models, we can optimize the accuracy overall.
Especially in the case of transfer learning, some models are able to recognize certain features better than others.  There are multiple ways to join models together.  One such way 
is simply taking the most popular vote of all the models in the ensemble.  Another better way is to weight certain model decisions for a certain class if that model tends to 
perform better on that class.

# Analysis
Claire did an excellent job explaining how we can easily build deep neural networks using Tensorflow Keras, taking advantage of Google's extremely powerful and easy to use library
for machine learning.  While at just an introductory level, I would have liked to see a more in depth explanation of what might go in a custom classifier following a pretrained model
in an architecture.  For some cases, it might be beneficial to add more dense layers, or add less dense layers.  Addressing how to avoid overfitting may have also been a good point
to make, since the example architectures lacked a dropout layer.  Overall, I found Claire's example with audience participation for the ensembling to be extremely powerful, and 
her on the fly decisions based off of the audience response further clarified the nature of CNNs and how they recognize images.  The roadRunner project is a fantastic project to 
explore CNNs when applied to satellite imagery - a slightly different dataset than traditional datasets.  Claire has had huge contributions to this project, and it's amazing to
see it get more recognition within the data science department.  Finally, I believe that the main takeaway from this talk is the emphasis on understanding concepts rather than having
raw coding ability.  The advantage and power that comes with using Tensorflow Keras is that implementation of these models is extremely easy, as long as the basic concepts are followed
and understood.  Claire did a great job here of framing building DNNs as both fun and exploratory within the framework of TFKeras.  This presentation was overall both informative and engaging.
