# Disaster
Packages to be installed
1)pandas
2)Numpy
3)Sklearn
4)Matplotlib
5)Seaborn
Dataset Seismic.csv is read from the file 
plot of count of Each label present in the UCI dataset is carried out
<html>
<body>
<img src='count.png'></img>
</body>
</html>
The categorical labels are converted to labels having the integer values
The dataset is divided into training and testing dataset
then the Randomforest algorithm is called using the sklearn package
the data is collected from the each tree of the randomforest and then fed to the artificial neural networks for prediction
the accuracy and loss are evaluated and plots are made based on that data.
<br>
<br>
<br>
<br>

The second_part is python file used for detecting the collapsed and surviving buildings using the satellite and aerial images of ABCD dataset present on github.
we converted the images to feedable format to the CNN architecture model using pandas and numpy having the three bands red,green and blue with no infrared band included in it.


we used Thingxworx for visualization and cloud to send the data massly.

