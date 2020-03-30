So, hello everybody, and welcome back to Practical Deep Learning for Coders. 
This is lesson two, and in the last lesson we started training our first models. 
We didn't really have any idea how that training was really working, but we were looking at a higher level at what was going on. 
And we learned about “What is machine learning?” and “How does that work?” and we realized that based on how machine learning worked that there are some fundamental limitations on what it can do, and we've talked about some of those limitations. 
And we also talked about how after you've trained a machine learning model, you end up with a program which behaves much like a normal program or something: with inputs and a thing in the middle and outputs. 
So today we're gonna finish up talking about that and we're going to then look at how we get those models into production and what some of the issues with doing that might be. 
I wanted to remind you that there are two sets of books--sorry two sets of notebooks--available to you. 
One is the fastbook repo (the full actual notebooks containing all the text of the O'Reilly book) and so this lets you see everything that I'm telling you in much more detail, and then as well as that there's the course v4 repo which contains exactly the same notebooks but with all the prose stripped away to help you study. 
So that's where you really want to be doing your experiment and your practice and so maybe as you listen to the video you can kind of switch back and forth between the video and reading or do one and then the other, and then put it away and have a look at the course v4 notebooks and try to remember like “Okay, what was this section about?” and run the code, and see what happens and change it and so forth. 
So we were looking at this line of code where we looked at how we created our data by passing in information--perhaps most importantly some way to label the data--and we talked about the importance of labeling. 
And in this case, this particular dataset whether it's a cat or a dog, you can tell by whether it's an uppercase or a lowercase letter in the first position. 
That's just how this dataset (that they tell you when the readme) works. 
And we also looked particularly at this idea of “valid percent equals 0.2,” and like “What does that mean? It creates a validation set.” and that was something I wanted to talk more about. 
The first thing I want to do though is point out that this particular labeling function returns something that's either true or false. 
And actually this data set as we'll see later also contains the actual breed of 37 different cat and dog breeds, so you can also grab that from the filename. 
In each of those two cases we're trying to predict a category “Is it a cat, or is it a dog?” or “Is it a German Shepherd, or a Beagle, or a Ragdoll cat, or whatever?” When you're trying to predict a category, so when the label is a category, we call that a classification model. 
On the other hand, you might try to predict how old is the animal, or how tall is it, or something like that, which is like a continuous number that could be like 13.2 or 26.5 or whatever. 
Anytime you're trying to predict a number, your label is a number you call that regression. 
Okay? So those are the two main types of model classification and regressions. 
This is very important jargon to know about. 
So the regression model attempts to predict one or more numeric quantities such as temperature, or location, or whatever. 
This is a bit confusing, because sometimes people use the word regression as a shortcut to a particular, for a… Like an abbreviation for a particular kind of model, called linear regression. 
That's super confusing, because that's not what regression means. 
Linear regression is just a particular kind of regression but I just wanted to warn you of that. 
When you start talking about regression a lot of people will assume you're talking about linear regression even though that's not what the word means. 
All right, so I wanted to talk about this valid percent zero point two thing. 
So as we described valid percent grabs, in this case, twenty percent of the data, if it's zero point two, and puts it aside like in a separate bucket and then when you train your model, your model doesn't get to look at that data at all. 
That data is only used to decide, to show you how accurate your model is. 
So if you train for too long, and or with not enough data, and/or a model with too many parameters, after a while the accuracy of your model will actually get worse, and this is called overfitting. 
Right? So we use the validation set to ensure that we're not overfitting. 
The next line of code that we looked at is this one, where we created something called a learner. 
We'll be learning a lot more about that, but a learner is basically, or is, something which contains your data and your architecture that is the mathematical function that you're optimizing, and so a learner is the thing that tries to figure out what are the parameters which best cause this function to match the labels in this data. 
So we’ll be talking a lot more about that, but basically this particular function ResNet34 is the name of a particular architecture which is just very good for computer vision problems. 
In fact the name really is ResNet and then 34 tells you how many layers there are. 
So you can use ones with bigger numbers here to get more parameters that will take to train, take more memory, more likely to overfit, but could also create more complex models. 
Right now though I wanted to focus on this part here which is metrics equals error rate. 
This is where you list the functions that you want to be the... 
That you want to be called with your data. 
With your validation data and print it out after each epoch, and epoch is what we call it when you look at every single image in the data set once. 
And so after you've looked at every image in the data set once we print out some information about how you're doing and the most important thing we print out is the result of calling these metrics so error rate is the name of a metric and it's a function that just prints out what percent of the validation set are being incorrectly classified by your model. 
So our metric is a function that measures the quality of the predictions using the validation set so error rates one another common metric is accuracy which is just 1 minus error rate so very important to remember from last week we talked about loss. 
Arthur Samuel had this important idea in machine learning that we need some way to figure out how good our how well our model is doing so that when we change the parameters we can figure out which set of parameters make that performance measurement get better or worse, that performance measurement is called the loss. 
The loss is not necessarily the same as your metric. 
The reason why is a bit subtle and we'll be seeing it in a lot of detail once we delve into the math in the coming lessons but basically you need a function you need a loss function where if you change the parameters by just a little bit up or just a little bit down you can see if the loss gets a little bit better or a little bit worse and it turns out that error rate and accuracy doesn't tell you that at all because you might change the parameters by such a small amount that none of your dog's predictions start becoming cats and none of your cat predictions start becoming dogs. 
So like your predictions don't change so your error rate doesn't change. 
Loss and metric are closely related but the metric is the thing that you care about the loss is the thing which your computer is using as the measurement of performance to decide how to update your parameters. 
So we measure overfitting by looking at the metrics on the validation set. 
So fast AI always uses the validation set to print out your metrics and overfitting is like the key thing that machine learning is about it's all about how do we find a model which fits the data not just for the data that we're training with but for data that the training algorithm hasn't seen before. 
So overfitting results when our model is basically “cheating”. 
A model can cheat by saying oh I've seen this exact picture before and I remember that that's a picture of a cat. 
So it might not have learnt what cats look like in general it just remembers you know that images one four and eight are cats and two and three and five are dogs and learns nothing actually about what they really look like. 
So that's the kind of cheating that we're trying to avoid we don't want it to memorize our particular data set. 
So we split off our validation data and what most of this are words you're seeing on the screen are from the book okay so I just copied and pasted them. 
So if we split off our validation data and make sure that our model never sees it during training, it's completely untainted by it so we can't possibly cheat. 
Not quite true! We can cheat, the way we could cheat is we could run we could fit a model look at the result and the validation set, change something a little bit fit another model look at the validation set change something a little bit we could do that like a hundred times until we find something with the validation set looks the best. 
But now we might have fit the validation set, right? 
So if you want to be really rigorous about this you should actually set aside a third bit of data called the test set that is not used for training and it's not used for your metrics. 
It's actually, you don't look at it until the whole project has finished. 
And this is what's used on competition platforms like Kaggle. 
On Kaggle, after the competition finishes your performance will be measured against a data set that you have never seen. 
And so, that's a really helpful approach and it's actually a great idea to do that like even if you're not doing the modeling yourself. 
So if you're if you're looking at vendors and you're just trying to decide today go with IBM or Google or Microsoft and they're all showing you how great their models are, what you should do is you should say, “Okay you go and build your models and I am going to hang on to 10% of my data and I'm not going to let you see it at all and when you're all finished, come back and then I'll run your model on the 10% of data you've never seen”. 
Now pulling out your validation and test sets is a bit subtle though. 
Here's an example of a simple little data set and this comes from a fantastic blog post that Rachel wrote that we will link to about creating effective validation sets. 
And you can see basically you have some kind of seasonal data set. 
Now if you just say, “Okay, fas.ai, I want to model that I want to create a my dataloader using a valid_percent of 0.2”, it would do this. 
It would delete randomly some of the dots, right? 
Now, this isn't very helpful because it's we can still cheat because these dots are right in the middle of other dots and this isn't what would happen in practice. 
What would happen in practice is we would want to predict this is sales by date right we want to predict the sales for next week. 
Not the sales for 14 days ago 18 days ago and 29 days ago, okay? 
So what you actually need to do to create an effective validation set here is not do it randomly but instead chop off the end, right? 
And so this is what happens in all Kaggle competitions pretty much that involve time, for instance, is the thing that you have to predict is the next like two weeks or so after the last data point that they give you and this is what you should do also for your test set so again if you've got vendors that you're looking at you should say to them okay after you're all done modeling we're going to check your model against data that is one week later than you've ever seen before. 
And you won't be able to retrain or anything because that's what happens in practice, right? Okay. 
There's a question, I've heard people describe overfitting as training error being below validation error does this rule of thumb end up being roughly the same as yours? 
Okay, so that's a great question. 
So, I think what they mean there is training loss versus validation loss. 
Because we don't print training error so we do print at the end of each epoch the value of your loss function for the training set and the value of the loss function for the validation set. 
And if you train for long enough, that's so so if it's training mostly your training loss will go down and your validation loss will go down. 
Because by definition, loss function is defined such as a lower loss function is a better model. 
If you start overfitting, your training loss will keep going down, right? 
Because like why wouldn't it? 
You know, you're getting better and better parameters. 
But your validation loss will start to go up because actually you started fitting to the specific data points in the training set and so it's not going to actually get better. 
It's going to get it's not going to get better for the validation set it'll start to get worse. 
However, that does not necessarily mean that you're overfitting or at least not overfitting in a bad way as we'll see it's actually possible to be at a point where the validation loss is getting worse but the validation accuracy or error or metric is still improving. 
So I'm not going to describe how that would happen mathematically yet because we need to learn more about loss functions but we will. 
But for now just realize that the important thing to look at is your metric getting worse, not your loss function getting worse. 
Thank you for that fantastic question. 
The next important thing we need to learn about is called transfer learning. 
So the next line of code said learn.fine_tune. 
Why does it say learn.fine_tune? 
Fine tune is what we do when we are transfer learning so transfer learning is using a pre-trained model for a task that is different to what it was originally trained for. 
So more jargon to understand our jargon. 
Let's look at that. 
What's a pre-trained model? 
So what happens is remember I told you the architecture we're using is called ResNet-34? 
So when we take that ResNet-34 that's just a just a mathematical function okay with lots of parameters that we're going to fit using machine learning. 
There's a big data set called ImageNet, that contains 1.3 million pictures of a thousand different types of thing, whether it be mushrooms or animals or airplanes or hammers or whatever. 
There's a competition or there used to be a competition that runs every year to see who could get the best accuracy on the ImageNet competition. 
And the models that did really well, people would take those specific values of those parameters and they would make them available on the internet for anybody to download. 
So if you download that you don't just have an architecture now you have a trained model. 
You have a model that can recognize a thousand categories of thing in images. 
Which probably isn't very useful unless you happen to want something that recognizes those exact thousand categories of thing. 
But it turns out you can rather you can start with those weights in your model and then train some more epochs on your data and you'll end up with a far far more accurate model than you would if you didn't start with that pre-trained model and we'll see why in just a moment, right?
 But this idea of transfer learning, it's kind of, it makes intuitive sense, right? 
ImageNet already has some cats and some dogs in it and it's you know it can say this is a cat and this is a dog, but you want to maybe do something that recognizes lots of breeds that aren't in ImageNet. 
Well, for it to be able to recognize cats versus dogs versus airplanes versus hammers it has to understand things like: what does metal look like? 
What does fur look like? 
What do ears look like? 
You know, so it can say like oh this breed of animal, this breed of dog has pointy ears and oh this thing is metal so it can't be a dog. 
So all these kinds of concepts get implicitly learned by a pre-trained model. 
So if you start with a pre-trained model then you don't have to learn all these features from scratch, and so transfer learning is the single most important thing for being able to use less data and less compute and get better accuracy. 
So that's a key focus for the fastai library and a key focus for this course. 
There's a question: I am a bit confused on the differences between loss, error, and metric. 
Sure, so error is just one kind of metric so there's lots of different possible labels you could have. 
Let's say you were trying to create a model which could predict how old a cat or dog is. 
So the metric you might use is: on average, how many years were you off by? 
So that would be a metric. 
On the other hand if you're trying to predict whether this is a cat or a dog your metric would be: what percentage of the time am I wrong? 
So that latter metric is called the error rate. 
Okay so error is one particular metric. 
It's a thing that measures how well you're doing and it's like it should be the thing that you most care about. 
So you write a function or use one of fastai's predefined ones which measures how well you're doing. 
Loss is the thing that we talked about in Lesson One so I'll give a quick summary but go back to lesson one if you don't remember. 
Arthur Samuel talked about how a machine learning model needs some measure of performance which we can look at: when we adjust our parameters up or down does that measure of performance get better or worse? 
And as I mentioned earlier, some metrics possibly won't change at all if you move the parameters up and down just a little bit. 
So they can't be used for this purpose of adjusting the parameters to find a better measure of performance. 
So quite often we need to use a different function we call this the loss function and the loss function is the measure of performance that the algorithm uses to try to make the parameters better and it's something which should kind of track pretty closely to the the metric you care about but it's something which, as you change the parameters a bit, the loss should always change a bit. 
And so there's a lot of hand waving there because we need to look at some of the math of how that works and we'll be doing that in the next couple of lessons. 
Thanks for their great questions. 
Okay so fine tuning is a particular transfer learning technique where the -- oh and you're still showing your picture and not the slides. 
So fine-tuning is a transfer learning technique where the weights (this is not quite the right word we should say the parameters) where the parameters of a pre-trained model are updated by training for additional epochs using a different task to that used for pre-training. 
So pre-training the task might have been ImageNet classification and then our different task might be recognizing cats versus dogs. 
So the way by default fastai does fine tuning is that we use one epoch, which, remember, is one looking at every image in the data set once. 
One epoch to fit just those parts of the model necessary to get the particular part of the model that's especially for your data set working. 
And then we use as many epochs as you asked for to fit the whole model. 
And so this is more if you for those people who might be a bit more advanced we'll see exactly how this works later on in the lessons. 
So why does transfer learning work, and why does it work so well? 
The best way in my opinion to look at this is to see this paper by Zeiler and Fergus, who were actually 2012 ImageNet winners and interestingly their key insights came from their ability to visualize what's going on inside a model. 
And so visualization very often turns out to be super important to getting great results. 
What they were able to do was they looked -- remember I told you like a resnet 34 has 34 layers? 
They looked at something called AlexNet which was the previous winner of the competition, which only had seven layers. 
At the time that was considered huge and so they took the seven layer model and they said what does the first layer of parameters look like? 
And they figured it out how to draw a picture of them right? 
And so the first layer had lots and lots of features but here are nine of them one two three four five six seven eight nine. 
And here's what nine of those pictures look like. 
One of them was something that could recognize diagonal lines from top left to bottom right. 
One of them could find diagonal lines from bottom left to top right. 
One of them could find gradients that went from the top of orange to the bottom of blue. 
Some of them were able you know, one of them was specifically for finding things that were green, and so forth right. 
So for each of these nine, they're called filters or features. 
So then something really interesting they did was they looked at each one of these, each one of these filters, each one of these features, and we'll learn kind of mathematically about what these actually mean in the coming lessons but for now, let's just recognize them and saying oh there's something that looks at diagonal lines and something that looks at gradients and they found in the actual images in imagenet specific examples of parts of photos that match that filter. 
So for this top left filter here are nine actual patches of real photos that match that filter and as you can see they're all diagonal lines. 
And so here's the for the green one here's parts of actual photos that match the green one. 
So layer one is super super simple and one of the interesting things to note here is that something that can recognize gradients and patches of color and lines is likely to be useful for lots of other tasks as well not just imagenet. 
So you can kind of see how something that can do this might also be good at many many other computer vision tasks as well. 
This is layer 2, layer 2 takes the features of layer 1 and combines them. 
so it can not just find edges that can find corners or repeating curving patterns or semi circles or full circles. 
And so you can see for example here's a, it's kind of hard to exactly visualize these layers after layer 1. 
You kind of have to show examples of what the filters look like. 
But here you can see examples of parts of photos that these, this layer 2 circular filter has activated on. 
And as you can see it's found things, with circles. 
So interestingly this one which is this kind of blotchy gradient seems to be very good at finding sunsets. 
And this repeating vertical pattern is very good at finding, like curtains and wheat fields and stuff. 
So the further we get, layer three then gets to combine all the kinds of features in layer two. 
And remember we're only seeing so anything here are twelve of the features but actually there's probably hundreds of them. 
I don't remember exactly in alex net but there's lots. 
But by the time we get to layer three by combining features from layer two it already has something which is finding text. 
So this is a feature which can find bits of image that contain text. 
It's already got something which can find repeating geometric patterns. 
And you see this is not just like a matching specific pixel patterns. 
This is like a semantic concept. 
It can find repeating circles or repeating squares or repeating hexagons. 
Great. 
So it's really like computing, it's not just matching a template. 
And remember we know that neural networks can solve any possible computable function. 
So it can certainly do that. 
So layer four gets to combine all the filters from layer three anyway at once. 
And so by layer four we have something that can find dog faces for instance. 
So you can kind of see how each layer we get like more applicatively more sophisticated features. 
And so that's why these deep neural networks can be so incredibly powerful. 
It's also why transfer learning can work so well. 
Because like, if we wanted something that can find books. 
And I don't think there's a book category in imagenet. 
Well it's actually already got something that can find text as an earlier filter which I guess it must be using to find maybe there's a category for library or something or a bookshelf. 
So when you use transfer learning you can take advantage of all of these pre-learnt features to find things that are as combinations of these or existing features. 
That's why transfer learning can be done so much more quickly and so much less data than traditional approaches. 
One important thing to realize then is that these techniques for computer vision are not just good at recognizing photos;  there's all kinds of things you can turn into pictures, for example these are sounds that have been turned into pictures by representing their frequencies over time and it turns out that if you convert a sound into these kinds of pictures you can get basically state-of-the-art results at sound detection just by using the exact same resnet learner that we've already seen. 
We need to highlight that it's 945 so if you want to take a break soon?
A really cool example from I think our very first year of running fastai; one of our students created pictures, they worked at Splunk in anti-fraud, and they created pictures of users moving their mouse and, if I remember correctly as they moved their mouse he basically drew a picture of where the mouse moved and the color depended on how fast they moved and these circular blobs is where they clicked the left or the right mouse button. 
At Splunk what he did actually for the course, as a project for the course, is he tried to see whether he could use this these pictures with exactly the same approach we saw in lesson 1 to create an anti-fraud model, and it worked so well that Splunk ended up patenting a new product based on this technique and you can actually check it out there's a blog post about it on the internet where they describe this breakthrough anti-fraud approach which literally came from one of our really amazing and brilliant and creative students after lesson one of the course. 
Another cool example of this is looking at different viruses and again turning them into pictures and you can kind of see how they've got here this is from a paper, check out the book for the citation, they've got three examples of a particular virus called VB.AT and another example of a particular virus called Fakerean and you can see in each case the pictures all look kind of similar and that's why again they can get state-of-the-art results in virus detection; by turning the program signatures into pictures and putting it through image recognition. 
So in the book you'll find a list of all of the terms, all of the most important terms, we've seen so far and what they mean I'm not going to read through them but I want you to please because these are the terms that we're going to be using from now on and you've got to know what they mean because if you don't you're going to be really confused because I'll be talking about labels and architectures and models and parameters and they have very specific exact meanings and I'll be using those exact meanings, so please review this. 
So to remind you this is where we got to; we ended up with Arthur Samuels overall approach and we replaced his terms with our terms so we have an architecture which contains parameters as inputs, well parameters and the data as inputs so that the architecture plus the parameters are the model, with the inputs they used to calculate predictions, they are compared to the labels with a loss function and that loss function is used to update the parameters many many times to make them better and better until the loss gets nice and super low. 
So this is the end of chapter 1 of the book. 
It's really important to look at the questionnaire because the questionnaire is the thing where you can check whether you have taken away from this book, this chapter the stuff that we hope you have. 
So go through it and anything that you're not sure about, the answer is in the text so just go back to earlier in the book and in the chapter you will find the answers. 
There's also a further research section after each questionnaire, for the first couple of chapters they're actually pretty simple hopefully they're pretty fun and interesting; they're things where to answer the question it's not enough to just look in the chapter, you actually have to go and do your own thinking and experimenting and googling and so forth. 
In later chapters some of these further research things are pretty significant projects that might take a few days or even weeks and so check them out because hopefully they'll be a great way to expand your understanding of the material. 
So something that Sylvain points out in the book is that if you really want to make the most of this then after each chapter please take the time to experiment with your own project and within the books we provide and then see if you can redo the notebooks on a new dataset. 
Perhaps for chapter one that might be a bit hard because we haven't really shown how to change things but for chapter two, which we're going to start next, you'll absolutely be able to do that. 
Okay so let's take a 5 minute break and we'll come back at 9:55 San Francisco time.
Okay so welcome back everybody and I think we've got a couple of questions to start with so Rachel please take it away. 
Sure, are filters independent by that I mean if filters are pre-trained might they become less good and detecting features of previous images when fine-tuned? 
Oh that is a great question, so assuming I understand the question correctly, if you start with say an imagenet model and then you fine-tune it on dogs versus cats for a few epochs and you get something that's very good at recognizing dogs versus cats it's going to be much less good as an imagenet model after that, so it's not going to be very good at recognizing aeroplanes or hammers or whatever. 
This is called catastrophic forgetting in the literature, the idea that as you see more images about different things to what you saw earlier that you start to forget what the things you saw earlier are. 
So if you want to fine-tune something which is good at a new task but also continues to be good at the previous task you need to keep putting in examples of the previous task as well. 
What are the differences between parameters and hyper parameters? 
If I am feeding an image of a dog as an input and then changing the hyper parameters of batch size in the model what would be an example of a parameter? 
So the parameters are the things that are described in lesson one that Arthur Samuel described as being the things which change what the model does, what the architecture does. 
So we start with this infinitely flexible function, the thing called a neural network, that can do anything at all and the way you get it to do one thing versus another thing is by changing its parameters. 
They are the numbers that you pass into that function so there's two types of numbers you pass into the function: there's the numbers that represent your input, like the pixels of your dog, and there's the numbers that represent their learnt parameters. 
So in the example of something that's not a neural net, but like a checkers playing program like Arthur Samuel might have used back in the early 60s and late 50s, those parameters may have been things like: if there is a opportunity to take a piece versus an opportunity to get to the end of a board how much more value should I consider one versus the other. 
You know it's twice as important or it's three times as important -- that two versus three -- that would be an example of a parameter. 
In a neural network, parameters are a much more abstract concept and so a detailed understanding of what they are will come in the next lesson or two, but it's the same basic idea: they’re the numbers which change what the model does to be something that recognizes malignant tumors, versus cats versus dogs versus colorizes black and white pictures. 
Whereas the hyperparameter is the choices about what numbers do you pass to the function, to the actual fitting function to decide how that fitting process happens. 
There's a question, “I'm curious about the pacing of this course. 
I'm concerned that all the material may not be covered.” Depends what you mean by all the material. 
We certainly won't cover everything in the world, so yeah we'll cover what we can. 
We’ll cover what we can in seven lessons; we're certainly not covering the whole book if that's what you're wondering. 
The whole book will be covered in either two or three courses. 
In the past it's generally been two courses to cover about the amount of stuff in the book but we'll see how it goes, because the book’s pretty big -- 500 pages. 
So when you say two courses, you mean fourteen lesson? Fourteen, yes it would be like 14 or 21 lessons to get through the whole book. 
Although having said that, by the end of the first lesson hopefully there'll be kind of like enough momentum and understanding that reading the book independently will be more useful and you'll have also kind of gained a community of folks on the forums that you can hang out with and ask questions of and so forth. 
So in in the second part of the course we're going to be talking about putting stuff in production and so to do that, we need to understand like what are the capabilities and limitations of deep learning? What are the kinds of projects that even make sense to try to put in production? 
And you know one of the key things I should mention in the book and in this course is that the first two or three lessons and chapters, there's a lot of stuff which is designed not just for the coders but for, for everybody. 
There's lots of information about, what are the practical things you need to know to make deep learning work. 
And so one of them, things you need to know is, “well what's deep learning actually good at at the moment?” 
So I'll summarize what the book says about this, but there are the kind of four key areas that we have as applications in Fastai: computer vision, text, tabular, and what I've called here “Recsys”, for recommendation systems and specifically a technique called collaborative filtering which we briefly saw... 
Sorry another question, are there any pre-trained weights available other than the ones from Imagenet that we can use? 
If yes, when should we use others and when Imagenet? 
Oh that's a really great question. 
So yes there are a lot of pre-trained models, and one way to find them.. 
And also you're currently just showing us.. 
Ok great. 
One great way to find them is you can look up models zoo which is a common name for places that have lots of different models. 
And so here's lots of models zoos. 
Or you can look for pre-trained models. 
And so yeah, there's quite a few, unfortunately not as wide a variety as I would like that most is still on Imagenet or similar kinds of general photos. 
For example medical imaging there's hardly any. 
There's a lot of opportunities for people to create domain-specific pre-trained models it's it's still an area that's really underdone because not enough people are working on transfer learning. 
Okay, so as I was mentioning we've kind of got these four applications that we've talked about a bit and deep learning is pretty, you know, pretty good at all of those tabular data like spreadsheets and database tables is an area where deep learning is not always the best choice but it's particularly good for things involving high cardinality variables, that means variables that have like lots and lots of discrete levels like zip code or product ID or something like that. 
Deep learning is really pretty great for those in particular. 
For text it's pretty great at things like classification and translation. 
It's actually terrible for conversation and so that's that's been something that's been a huge disappointment for a lot of companies I tried to create these like conversation bots, but actually deep learning isn't good at providing accurate information it's good at providing things that sound accurate and sound compelling but that we don't really have great ways yet of actually making sure it's correct. 
One big issue for recommendation systems collaborative filtering is that deep learning is focused on making predictions which don't necessarily actually mean creating useful recommendations. 
We'll see what that means in a moment. 
Deep learning is also good at multimodal that means things where you've got multiple different types of data so you might have some tabular data including a text column and an image, then some collaborative filtering data and combining that all together is something that deep learning is really good at. 
So for example putting captions on photos is something which deep learning is pretty good at, although again, it's not very good at being accurate. 
So what you know might say this is a picture of two birds when it's actually a picture of three birds and then this other category there's lots and lots of things that you can do with deep learning by being creative about the use of these kinds of other application based approaches, for example an approach that we developed for natural language processing called ULMFit that we will be learning in the course. 
It turns out that it's also fantastic you're doing protein analysis. 
If you think of the different proteins as being different words and they're in a sequence which has some kind of state and meaning it turns out that ULMFit works really well for protein analysis. 
So often it's about kind of being being creative. 
So to decide like for the product that you're trying to build is deep learning gonna work well for it, in the end you kind of just have to try it and see but if you if you do a search you know hopefully you can find examples about the people that have tried something similar even if you can't that doesn't mean it's not going to work. 
So for example I mentioned the collaborative filtering issue where a recommendation and a prediction are not necessarily the same thing. 
You can see this on Amazon for example quite often. 
So I bought a Terry Pratchett book and then Amazon tried for months to get me to buy more Terry Pratchett books. 
Now that must be because their predictive model said that people who bought one particular Terry Pratchett book are likely to also buy a other Terry Pratchett books. 
But from the point of view of like well is this going to change my buying behavior: probably not, right, like if I liked that book I already know I like that author and I already know that like they probably wrote other things so I'll go and buy it anyway. 
So this would be an example of like Amazon probably not being very smart, up here they're actually showing me collaborative filtering predictions rather than actually figuring out how to optimize a recommendation. 
So an optimized recommendation would be something more like your local human bookseller might do, where they might say, “Oh! you like Terry Pratchett, well let me tell you about other kind of comedy fantasy sci-fi writers on the similar vein who you might not have heard about before”. 
So the difference between recommendations and predictions is super important. 
So I wanted to talk about a really important issue around interpreting models and for a case study for this I thought we let's pick something that's actually super important right now which is a model in this paper. 
One of the things we're going to try and do in this course is learn how to read papers. 
So here is a paper which you I would love for everybody to read called high temperature and high humidity reduce the transmission of COVID-19. 
Now this is a very important issue because if the claim of this paper is true then that would mean that this is going to be a seasonal disease and if this is a seasonal disease and it's going to have massive policy implications. 
So let's try and find out how this was modeled and understand how to interpret this model. 
So this is a key picture from the paper and what they've done here is they've taken a hundred cities in China and they've plotted the temperature on one axis, in Celsius, and R on the other axis, where R is a measure of transmissibility. 
It says for each person that has this disease how many people on average will they infect. 
So if R is under 1, then the disease will not spread. 
If R is higher than like 2 it's going to spread incredibly quickly. 
Basically R is going to, you know, any high R is going to create an exponential transmission impact. 
And you can see in this case they have plotted a best fit line through here. 
Then they've made a claim that there's some particular relationship in terms of a formula that R is 1.99 minus 0.023 times temperature. 
So very obvious concern I would have looking at this picture is that this might just be random, maybe there's no relationship at all but just if you picked a hundred cities at random perhaps they were sometimes show this level of relationship. 
So one simple way to kind of see that would be to actually do it in a spreadsheet. 
So here is a spreadsheet. 
What I did was I kind of eyeballed this data and I guessed what is the mean degrees centigrade. 
I think it's about 5. 
What about the standard deviation of centigrade. 
I think it's probably about 5 as well. 
And then I did the same thing for R. 
I think the mean R looks like it's about 1.9 to me. 
And it looks like the standard deviation of R is probably about 0.5. 
So what I then did was I just jumped over here and I created a random normal value, so a random value from a normal distribution, so a bell curve, with that particular mean and standard deviation of temperature and that particular mean and standard deviation of R. 
And so this would be an example of a city that might be in this data set of a hundred cities. 
Something with 9 degrees Celsius and R of 1.1; so that would be 9 degrees Celsius and R of 1.1, something about here. 
So then I just copied that formula down 100 times. 
So here are a hundred cities that could be in China right, where this is assuming that there is no relationship between temperature and R right. 
They are just random numbers and so each time I recalculate that so if I hit control equals it will just recalculate it right. 
I get different numbers okay because they're random. 
And so you can see at the top here I've then got the average of all of the temperatures and the average of all of the R and the average of all the temperatures varies and the average of all of the R varies as well. 
So then what I did was I copied those random numbers over here. 
let's actually do it. 
So I'll go copy these 100 random numbers and paste them here here here here. 
And so now I've got 1 2 3 4 5 6 I've got 6 kind of groups of 100 cities. 
All right and so let's stop those from randomly changing any more by just fixing them in stone there.
Okay, so now that I've pasted them in, I've got 6 examples of what a hundred cities might look like if there was no relationship at all between temperature and R. 
I've got their mean temperature and R in each of those six examples. 
What I've done, is you can see here, at least for the first one, is I've plotted it, right?  
You can see, in this case, there's actually a slight positive slope. 
I've actually calculated the slope for each, just by using the slope function in Microsoft Excel. 
You can see that actually, in this particular case, is just random - five times it's been negative, and it's even more negative than their 0.023. 
So you can like, it's kind of matching our intuition here, which is that the slope of the line that we have here, is something that absolutely can often happen totally by chance. 
It doesn't seem to be indicating any kind of real relationship at all. 
If we wanted that slope to be more confident, we would need to look at more cities. 
Here I've got 3,000 randomly generated numbers. 
You can see here the slope is 0.00002, right? 
It's almost exactly zero, which is what we'd expect, when there's actually no relationship between C and R, and in this case there isn't - they're all random . 
Then if we look at lots and lots of randomly generated cities, then we can say, oh yeah, there's no slope. 
But when you only look at a hundred, as we did here, you're going to see relationships totally coincidentally, very, very often. 
So that's something that we need to be able to measure. 
One way to measure that is we use something called a p-value. 
A p-value, here's how a p-value works: we start out with something called a null hypothesis. 
The null hypothesis is basically what's our starting point assumption. 
Our starting point assumption might be, oh there's no relationship between temperature and R. 
And then we gather some data and (Rachel: have you explained what R is?) I have, yes. 
R is the transmissibility of the virus. 
So then we gather data of independent and dependent variables - in this case the independent variable is the thing that we think might cause the dependent variable. 
Here the independent variable would be temperature, the dependent variable would be R. 
So here we've gathered data - there's the data that was gathered in this example, and then we say what percentage of the time would we see this amount of relationship, which is a slope of 0.023 by chance? 
And as we've seen, one way to do that is by, what we would call, a simulation, which is by generating random numbers - a 100 set pairs of random numbers, a bunch of times, and seeing how often you see this relationship. 
We don't actually have to do it though. 
There's actually a simple equation we can use to jump straight to this number, which is, what percent of the time would we see that relationship by chance? 
And this is basically what that looks like. 
We have the most likely observation, which in this case would be if there is no relationship between temperature. 
Then the most likely slope would be zero, and sometimes you get positive slopes by chance, and sometimes you get pretty small slopes, and sometimes you get large negative slopes by chance. 
And so, the larger the number, the less likely it is to happen, whether it be on the positive side or the negative side. 
In our case, our question was - how often are we going to get less than negative 0.023? 
It would actually be somewhere down here. 
I actually copy this from Wikipedia, where they were looking for positive numbers, and so they've colored in this area above the number. 
This is the p-value, and we don't care about the math but there's a simple little equation you can use to directly figure out this number - the p-value -  from the data. 
This is kind of how nearly all kind of medical research results tend to be shown, and folks really focus on this idea of p-values. 
And indeed, in this particular study as we’ll see in a moment, they reported p-values. 
Probably a lot of you have seen p-values in your previous lives. 
They come up in a lot of different domains. 
Here's the thing - they are terrible. 
You almost always shouldn't be using them. 
Don't just trust me. 
Trust the American Statistical Association. 
They point out six things about p-values, and those include: p-values do not measure the probability that the hypothesis is true, or,  the probability that the data were produced by random choice alone. 
Now we know this because we just saw that, if we use more data, if we sample three thousand random cities rather than a hundred, we get a much smaller value. 
So p-values don't just tell you about how big a relationship is, but they actually tell you about a combination of that, and, how much data did you collect. 
So they don't measure the probability that the hypothesis is true. 
So therefore, conclusions and policy decisions should not be based on whether a p-value passes some threshold. 
P-value does not measure the importance of a result, because, again, it could just tell you that you collected lots of data, which doesn't tell you that the results are actually of any practical import. 
By itself, it does not provide a good measure of evidence. 
Frank Harrell, who is somebody whom I read his book, and it's a really important part of my learning. 
He's a professor of biostatistics, has a number of great articles about this. 
He says null hypothesis testing and p-values have done significant harm to science. 
He wrote another piece called “null hypothesis significance testing never worked”. 
I've shown you what p-values are so that you know why they don't work, not so that you can use them. 
But they're a super important part of machine learning because they come up all the time. 
When people are saying, this is how we decide whether your drug worked, or whether there is an epidemiological relationship, or whatever. 
And indeed, p-values appear in this paper. 
In the paper, they show the results of a multiple linear regression. 
They put three stars next to any relationship which has a p-value of 0.01 or less. 
There is something useful to say about a small p-value, like 0.01 or less. 
Which is the thing that we're looking at did not, probably did not happen by chance, right? 
The biggest statistical error people make all the time is that they see that a p-value is not less than 0.05 and then they make the erroneous conclusion that no relationship exists, right? 
Which doesn't make any sense because like let's say you only had like three data points then you almost certainly won't have enough data to have a p-value of less than 0.05 for any hypothesis. 
So like the way to check, is to go back and say, what if I picked the exact opposite null hypothesis? 
What if my null hypothesis was there is a relationship between temperature and R? 
Then do I have enough data to reject that null hypothesis, alright? And if the answer is no, then you just don't have enough data to make any conclusions at all, alright? 
So in this case they do have enough data to be confident that there is a relationship between temperature and R. 
Now that's weird because we just looked at the graph, and we did a little back of a bit of a back-of-the-envelope in Excel and we thought this is, could it, could well be random. 
So here's where the issue is. 
The graph shows what we call a univariate relationship. 
A univariate relationship shows the relationship between one independent variable and one dependent variable, and that's what you can normally show on a graph. 
But in this case they did a multivariate model in which they looked at temperature, and humidity, and GDP per capita, and population density, and when you put all of those things into the model then you end up with statistically significant results for temperature and humidity. 
Why does that happen? 
Well the reason that happens is because all these variations in the blue dots, is not random. 
There's a reason they're different, right? 
And the reasons include, denser cities are going to have higher transmission, for instance, and probably more humid will have less transmission. 
So when you do a multivariate model, it actually allows you to be more confident of your results, right? 
But the p-value as noted by the American Statistical Association does not tell us whether this is of practical importance. 
The thing that tells us if this is of practical as importance, is the actual slope that's found. 
And so in this case the equation they come up with is that R = three point nine six eight minus three point O point O three eight by temperature minus point zero two four by relative humidity this is this equation is this practically important. 
Well we can again do a little back of the envelope here, by just putting that into Excel. 
Let's say there was one place it had a temperature of ten centigrade and a humidity of forty, then if this equation is correct R would be about two point seven somewhere with the temperature of 35 centigrade and a humidity of eighty will be about point eight. 
So is this practically important? Oh my god yes, right? 
Two different cities, with different climates can be, if they're the same in every other way, and this model is correct then one city would have no spread of disease (because R is less than 1), one would have massive exponential explosion. 
So we can see from this model that if the modeling is correct, then this is a highly practically significant result. 
So this is how you determine practical significance of your models is not with p-values but with looking at kind of actual outcomes. 
So how do you think about the practical importance of a model and how do you turn a predictive model into something useful in production.
So I spent many many years thinking about this, and I actually created a with some other great folks actually created a paper about it.
"Designing great data products" 
And this is largely based on ten years of work I did at a company I founded called Optimal Decisions Group. 
And Optimal Decisions Group was focused on the question of helping insurance companies figure out what prices to set.
And insurance companies up until that point had focused on predictive modeling.
Actuaries, in particular, spent their time trying to figure out how likely is it that you're going to crash your car and if you do how much damage might you have and then based on that try to figure out what price they should set for your policy. 
So for this company, what we did was we decided to use a different approach which I ended up calling the drivetrain approach which is described here to set insurance prices and indeed to do all kinds of other things.
And so, for the Insurance example, the objective would be for an insurance company would be how do I maximize my, let's say, five-year profit.
And then, what inputs can we control can we control which what I call levers - so in this case it would be what price can I set.
And then data is data which can tell you as you change your levers how does that change your objective.
So if I start increasing my price to people who are likely to crash their car, then we will get less of them which means we have less costs, but at the same time, we'll also have less revenue coming in, for example.
So to link up there kind of the levers to the objective via the data we collect, we build models that described how the levers influence the objective.
And this is all like it seems pretty obvious when you say it like this but when we started work with Optimal Decisions in 1999, nobody was doing this in insurance,
Everybody in insurance was simply doing a predictive model to guess how likely people were to crash their car, and then pricing was set by like adding 20% or whatever.
It was just done in a very kind of naive way.
So what I did is I, you know, over many years took this basic process and tried to help lots of companies figure out how to use it to turn predictive models into actions.
So the starting point in like actually getting value in a particular model is thinking about what is it you're trying to do, and you know what are the sources of value in that thing you're trying to do.
The levers - what are the things you can change? 
Like what's the point of a predictive model if you can't do anything about it, right? 
Figuring out ways to find what data you you don't have, which ones suitable, what's available, then thinking about what approaches to analytics you can then take.
And then super important, like well, can you actually implement, you know, those changes. 
And super super important how do you actually change things as the environment changes. 
And, you know, interestingly a lot of these things are areas where there's not very much academic research. 
There's a little bit. 
And some of the papers that have been particularly around “maintenance” of like; How do you decide when your machine learning model is kind of still okay? 
How do you update it over time? 
Have had like many many many many citations, but they don't pop up very often because a lot of folks are so focused on the math. 
You know. 
And then there's the whole question of like “What constraints are in place across this whole thing?” So what you'll find in the book, is there is a whole appendix which actually goes through every one of these six things. 
And has a whole list of examples. 
So this is an example of how to like think about value. 
And lots of questions that companies and organizations can use to try and think about, you know, all of these different pieces of the actual puzzle of getting stuff into production and actually into an effective product. 
We have a question. 
Sure, just a moment. 
So I was going to say, so do check out this appendix because it actually originally appeared as a blog post and I think, except for my covid-19 posts that I did with Rachel, it's actually the most popular blog post I've ever written. 
It’s had hundreds of thousands of views. 
And it kind of represents like 20 years of hard won insights about like how you actually get value from machine learning and practice and what you actually have to ask. 
So please check it out because hopefully you'll find it helpful. 
So when we think about like think about this for the question of how should people think about the relationship between seasonality and transmissibility of covid-19, you kind of need to dig really deeply into the questions about like oh not just what what's that what are those numbers in the data, but what does it really look like right. 
So one of the things in the paper that they show is actual maps, right of temperature and humidity and R right. 
And you can see like, not surprisingly, that humidity and temperature in China are what we would call auto-correlated. 
Which is to say that places that are close to each other, in this case geographically, have similar temperatures and similar humidities. 
And so like this actually puts into the question a lot the p values that they have right. 
Because you can't really think of these as a hundred totally separate cities. 
Because the ones that are close to each other probably have very close behavior so maybe you should think of them as like a small number of sets of cities, you know of kind of larger geographies. 
So these are the kinds of things that when you look actually into a model you need to like think about what are, what are the limitations?  
But then to decide like well, what does that mean? 
What do I what do about that? 
You need to think of it from this kind of utility point of view, this kind of end to end, what are the actions I can take? 
What are the order the results point of view?  
Not just null hypothesis testing. 
So in this case for example there are basically four possible key ways this could end up. 
It could end up that there really is a relationship between temperature and R, or so that's but the right hand side is. 
Or there is no real relationship between temperature and R. 
And we might act on the assumption that there is a relationship. 
Or we might act on the assumption that there isn't a relationship. 
And so you kind of want to look at each of these four possibilities and say like well what would be the economic and societal consequences?  
And you know there's gonna be a huge difference in lives lost and you know economies crashing and whatever else - you know for each of these four. 
The paper actually you know has shown, if their model is correct, what's the likely R value in March for like every city in the world. 
And the likely R value in July for every city in the world. 
And so for example if you look at kind of New England and New York, the prediction here is and also West, the other the very coast of the west coast is that in July the disease will stop spreading. 
Now you know if that happens, if they're right then, that's gonna be a disaster because I think it's very likely in America and also the UK, that people will say “Oh turns out this disease is not a problem you know. 
It didn't really take off at all. 
The scientists were wrong.” People will go back to their previous day-to-day life and we could see what happened in 1918 flu virus of like the second go around. 
When winter hits could be much worse than the start right. 
So like there's these kind of like huge potential policy impacts depending on whether this is true or false. 
And so to think about it. 
Yes? I also just wanted to say that it would be it would be very irresponsible to think “oh summer’s gonna solve it. 
We don't need to act now.”  Just in that this is something growing exponentially and could do a huge huge amount of damage. 
Yeah yes okay. 
It already has done by the way. 
If you assume that there will be seasonality and that summer will fix things then it could lead you to be apathetic now. 
If you assume there's no seasonality and then there is, then you could end up kind of creating a larger level of expectation of destruction that actually happens and end up with your population being even more apathetic you know so that they're you know. 
Being wrong in any direction could be a problem. 
So one of the ways we tend to deal with this, with with this kind of modeling is we try to think about priors. 
So our priors are basically things where we, you know rather than just having a null hypothesis, we try and start with a guess as to like well what's what's more likely?  
Right so in this case if memory serves correctly I think we know that like flu viruses become inactive at 27 centigrade we know that like cold, the cold coronaviruses are seasonal. 
The 1918 flu pandemic was seasonal. 
In every country and city that’s been studied so far, there's been quite a few studies like this. 
They've always found climate relationships so far. 
So maybe we'd say: “Well prior belief is that this thing is probably seasonal.” And so then we’d say: “Well this particular paper adds some evidence to that.” So it shows how incredibly complex it is to use a model in practice for in this case policy discussions but also for organizational decisions. 
Because, you know, there's always complexities, there's always uncertainties. 
And so you actually have to think about the utilities, you know. 
And your best guesses and try to combine everything together as best as you can. 
Okay. 
So with all that said. 
It's still nice to be able to get our models up and running even if, you know - even just a predictive model is sometimes useful on its own. 
Sometimes it's useful to prototype something, and sometimes it's got to be part of some bigger picture. 
So rather than try to create some huge end-to-end model here. 
We thought we would just show you how to get your Pytorch FastAI model up-and-running. 
In as raw a form as possible. 
So that from there, you can kind of build on top of it, as you like. 
So to do that; we are going to download and curate our own dataset. 
And you're going to do the same thing. 
You're going to train your own model, on that dataset, and then you're going to create an application, and then you're going to host it. 
Right? Now, there're lots of ways to curate an image dataset; you might have some photos on your own computer, there might be stuff at work you can use. 
One of the easiest though, is just to download stuff off the internet. 
There’s lots of services for downloading stuff off the internet. 
We're going to be using Bing Image Search here. 
Because they're super easy to use. 
A lot of the other kind of easy to use things require breaking the Terms of Service of websites. 
So we're not going to show you how to do that. 
But there’s lots of examples that do show you how to do that. 
So you can check them out as well, if you want to. 
Bing Image Search is actually pretty great at least at the moment. 
These things change a lot, so keep an eye on our website to see if we've changed our recommendation. 
The biggest problem with Bing Image Search is that the signup process is a nightmare, at least at the moment. 
One of the hardest parts of this book is just signing up to their damn API. 
Which requires going through Azure. 
It's called Cognitive Services - Azure Cognitive Services. 
So we'll make sure that all that information is on the website for you to follow through just how to sign up. 
So we're going to start from the assumption that you've already signed up. 
But you can find it, just go: Bing, Bing Image Search API. 
And at the moment they give you seven days with a pretty high quota for free. 
And then after that, you can keep using it as long as you like but they kind of limit it to like three transactions per second or something. 
Which is still plenty. 
You can still do thousands for free so it's at the moment it's pretty great even for free. 
So what will happen is when you sign up for Bing Image Search, or any of these kind of services, they'll give you an API key. 
So just replace the ‘XXX’ here with the API key that they give you. 
Okay, so that's now going to be called “key”. 
In fact, let's do it over here. 
Okay, so you'll put in your key and then there's a function we've created called search_images_bing which is just a super tiny little function. 
As you can see, it's just two lines of code -- I was just trying to save a little bit of time,  which will take some take your API key and some search term and return a list of URLs that match that search term. 
As you can see for using this particular service you have to install a particular package,  so we show you how to do that on the site as well. 
So once you've done so you'll be able to run this and that will return by default I think 150 URLs. 
Okay, so fast.ai comes with a download_url function, so let's just download one of those images just to check and open it up. 
And so what I did was I searched for “grizzly bear” and here I have a grizzly bear. 
So then what I did was I said, okay, let's try and create a model that can recognize grizzly bears versus black bears versus teddy bears, so that way I can find out. 
I could set up some video recognition system near our campsite when we're out camping that gives me bear warnings, but if it's a teddy bear coming then it doesn't warn me and wake me up,  because that would not be scary at all. 
So then I just go through each of those three bear types, create a directory with the name of grizzly or black or teddy bear searched Bing for that particular search term along with bear and download. 
And so download_images is a fast.ai function as well. 
So after that I can call get_image_files which is a fast.ai function that will just return recursively all of the image files inside this path. 
And you can see it's given me bears/black/ and then lots of numbers. 
So one of the things you have to be careful of is that a lot of the stuff you download will turn out to be like not images at all and will break. 
So you can call verify_images to check that all of these file names are actual images. 
And in this case I didn't have any failed, so there's it's empty. 
But if you did have some, then you would call Path.unlink to unlink. 
Path.unlink is part of the Python standard library and it deletes a file. 
And map is something that will call this function for every element of this collection. 
This is part of a special fast.ai class called “L”. 
It’s basically it's kind of a mix between the Python standard library list class and a numpy array class.
Then we'll be learning more about it later in this course, but it basically tries to make it super easy to do kind of more functional-style programming in Python. 
So in this case it's going to unlink everything that's in the failed list, which is probably what we want now, because there are all the images that fail to verify. 
All right,  so we've now got a path that contains a whole bunch of images and they're classified according to black, grizzly, or teddy, based on what folder they're in. 
and so to create so we're going to create a model. 
and so to create a model the first thing we need to do is to tell fast.ai what kind of data we have and how it’s structured. 
Now in part in Lesson 1 of the course we did that by using what we call a factory method which is we just said image_data_loader start from name, and it did it all for us. 
Those factory methods are fine for beginners, but now we're into Lesson 2. 
We're not quite beginners anymore, so we're going to show you the super super flexible way to use data in whatever format you like, and it's called the DataBlock API. 
And so the DataBlock API looks like this. 
Here's the DataBlock API. 
You tell fast.ai what your independent variable is and what your dependent variable is. 
So what your labels are and what your input data is. 
So in this case our input data are images and our labels are categories. 
So category is going to be either grizzly, or black, or teddy. 
So that's the first thing you tell it. 
Now that's the block's parameter. 
And then you tell it - how do you get a list of all of the, in this case file names, right. 
And we just saw how to do that because we just called the function ourselves. 
The function is called get_image_files. 
So we tell it what function to use to get that list of items and then you tell it -  how do you split the data into a validation set and a training set. 
And so we're going to use something called a RandomSplitter which just splits it randomly. 
And we're going to point 30% of it into the validation set. 
We're also going to set the random seed which ensures that every time we run this, the validation set will be the same. 
And then you say, okay, how do you label the data. 
And this is the name of a function called parent_label. 
And so that's going to look for each item at the name of the parent. 
So this, this particular one would become a black bear. 
Now this is like the most common way for image datasets to be represented, is that they get put the different images get the files get put into folder according to their label. 
And then finally here we've got something called item_tfms. 
We'll be learning a lot more about transforms in a moment. 
That these are basically functions that get applied to each image. 
And so each image is going to be resized to 128 by 128 square. 
So we're going to be learning more about DataBlock API soon. 
But basically the process is going to be -- it's going to call whatever is get_items, which is a list of image files. 
And then it’s going to call get_x,  get_y so in this case  there's no get_x but there is a get_y so it's just parent label. 
And then it's going to call the create method for each of these two things - it's going to create an image and it's going to create a category. 
And so I'm going to call the item_tfms,  which is resize. 
And then the next thing it does is it puts it into something called a data loader. 
A data loader is something that grabs a few images at a time (I think by default it’s 64) and puts them all into a single, it's called a batch. 
It just grabs 64 images and sticks them all together. 
And the reason it does that is it then puts them all onto the GPU at once so it can pass them all to the model through the GPU in one go. 
And that's going to let the GPU go much faster, as we'll be learning about. 
And then finally (we don't use any here), we can have something called batch transforms, which we will talk about later. 
And then somewhere in the mineral about here conceptually is the splitter which is the thing that splits into the training set and the validation set. 
So this is a super flexible way to tell fast.ai how to work with your data. 
And so at the end of that it returns an object of type DataLoaders. 
That's why we always call these things DLs, right. 
So, DataLoaders has a validation and a training DataLoader. 
And a DataLoader as I just mentioned is something that grabs a batch of a few items at a time and puts it on the GPU for you. 
So this is basically the entire code of DataLoaders. 
So the details don't matter, I just wanted to point out that like a lot of these concepts in fast.ai, when you actually look at what they are, they’re incredibly simple little things. 
It's literally something that you just pass in a few data loaders to and it stores them in an attribute. 
And pass and gives you the first one back as .train and second one back as .valid. 
So we can create our DataLoaders by first of all creating the DataBlock ,and then we call the DataLoaders, passing in our path to create DLs. 
And then you can call show_batch on that. 
You can call show_batch pretty much anything in fast.ai to see your data. 
And look, we've got some grizzlies, we've got a teddy, we've got a grizzly. 
So you get the idea right. 
I'm going to look at these different, I'm going to look at data augmentation next week so, I'm going to skip over data augmentation and let's just jump straight into trading your model. 
So once we've got DLs, we can just like in Lesson 1, call cnn_learner to create a ResNet. 
We’'re going to create a smaller ResNet this time, a ResNet18. 
Again, asking for error rate, we can then call .fine_tune again. 
So you see it's all the same lines of code we've already seen. 
And you can see our error rate goes down from nine to one, so we've got 1% error and after training for about 25 seconds. 
So you can see you know we've only got 450 images we've trained for well less than a minute and we only have let's look at the confusion matrix so we can say, “I want to create a classification interpretation class; I want to look at the confusion matrix” and the confusion matrix. 
As you can see, it's something that says “for things that are actually black bears, how many are predicted to be black bears versus grizzly bears versus teddy bears?”  
So, the diagonal are the ones that are all correct and so it looks like we've got two errors. 
We've got one grizzly that was predicted to be black and one black that was predicted to be grizzly. 
A super, super useful method is “plot top losses” and that'll actually show me what my errors actually look like. 
So, this one here was predicted to be a grizzly bear but the label was “black bear”. 
This one was the one that's predicted to be a black bear and the label was “grizzly bear”. 
These ones here are not actually wrong. 
This is predicted to be “black” and it's actually black. 
But, the reason they appear in this is because these are the ones that the model was the least confident about. 
Okay, so we're going to look at the image classifier cleaner next week. 
Let's focus on how we then get this into production. 
So, to get it into production, we need to export the model. 
So, what exporting the model does is that it creates a new file, which by default is called “export.pkl”, which contains the architecture and all of the parameters of the model. 
So, that is now something that you can copy over to a server somewhere and treat it as a predefined program, right? 
So, then the process of using your trained model on new data kind of in production is called “inference”. 
So, here I've created an inference learner by loading that learner back again, all right, and so obviously it doesn't make sense to do it right next to after I've saved it in a notebook. 
But, I'm just showing you how it would work right. 
So, this is something that you would do on your server- inference. 
Remember that once you have trained a model, you can just treat it as a program- you can pass inputs to it. 
So, this is now our program. 
This is our bear predictor. 
So, I can now call “predict” on it and I can pass it an image and it will tell me- here it is 99.999% sure- that this is a “grizzly”. 
So, I think what we're going to do here is we're going to wrap it up here and next week we'll finish off by creating an actual GUI for our bear classifier. 
We will show how to run it for free on a service called “Binder” and, yeah, and then I think we'll be ready to dive into some of the details of what's going on behind the scenes. 
Any questions or anything else before we wrap up, Rachel?  No. 
Okay, great. 
All right, thanks everybody. 
So, we hopefully, yeah, I think from here on we've covered, you know, most of the key kind of underlying foundational stuff from a machine-learning point of view that we're going to need to cover. 
So, we'll be able to, ready to dive into lower-level details of how deep learning works behind the scenes and I think that'll be starting from next week. 
So, see you then.