# Machine learning on spectral data

- Spectral data obtained from a spectrometer from sampling a liquid (details are witheld due to confidentiality)
- Project aims to predict test results for 'M' 'F' 'P' or 'C' using the spectral properties of the sample
- All results are continuous variables - this is a regression problem
- Since spectral data across 2047 wavelengths between 336-1015 nm, and there are a total of 150 samples (x6 reads), this is a very wide dataset -> data reduction is required
- Since variation of wavelength intensity across the spectrum >> variation between samples at given wavelength, some standardisation is required
- Current data is 1/6 of total sample: this notebook only aims to generate candidate models for future testing

### Feature preprocessing

- Converted values to z-score for each wavelength, essentially flattening the shape of the spectral signature
- Sampled only between 400 nm and 822 nm (1230 points)
- Took a mean across each 10 points (123 points)

### Models attempted

- Artificial neural networks built using Keras Sequential() with Dense() connetions
- Lasso regression
- Support Vector Regression
- Random Forest Regression

### Notes: 24/12/2017

- Model generated for 'P', try other dependent variables next
- Can SVM model be applied to another test dataset? 
- Can performance of neural net be enhanced by a larger training dataset (by combining with the other data)?