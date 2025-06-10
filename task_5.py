# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Display all columns
pd.set_option('display.max_columns', None)

# Load the dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic = pd.read_csv(url)

# Display first 5 rows
print(titanic.head())

# Basic information about the dataset
print("\nDataset Info:")
print(titanic.info())

# Statistical summary
print("\nStatistical Summary:")
print(titanic.describe(include='all'))

# Check for missing values
print("\nMissing Values:")
print(titanic.isnull().sum())

# Handle missing values in Age (though your dataset shows no missing values)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# Create new features
titanic['FamilySize'] = titanic['Siblings/Spouses Aboard'] + titanic['Parents/Children Aboard'] + 1
titanic['IsAlone'] = 0
titanic.loc[titanic['FamilySize'] == 1, 'IsAlone'] = 1

# Age Bins
titanic['AgeBin'] = pd.cut(titanic['Age'], bins=[0, 12, 20, 40, 60, 100], 
                          labels=['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior'])

# Fare Bins
titanic['FareBin'] = pd.qcut(titanic['Fare'], 4, labels=['Low', 'Mid', 'High', 'Very High'])

# Verify changes
print("\nAfter Feature Engineering:")
print(titanic.info())

# Visualization 1: Survival Count
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.countplot(x='Survived', data=titanic)
plt.title('Survival Count')

# Visualization 2: Survival by Class
plt.subplot(1, 3, 2)
sns.countplot(x='Pclass', hue='Survived', data=titanic)
plt.title('Survival by Passenger Class')

# Visualization 3: Survival by Gender
plt.subplot(1, 3, 3)
sns.countplot(x='Sex', hue='Survived', data=titanic)
plt.title('Survival by Gender')

plt.tight_layout()
plt.show()

# Visualization 4: Age Distribution by Survival
plt.figure(figsize=(12, 6))
sns.histplot(data=titanic, x='Age', hue='Survived', element='step', stat='density', common_norm=False)
plt.title('Age Distribution by Survival Status')
plt.show()

# Visualization 5: Fare Distribution by Survival
plt.figure(figsize=(12, 6))
sns.boxplot(x='Survived', y='Fare', data=titanic)
plt.title('Fare Distribution by Survival Status')
plt.yscale('log')  # Using log scale due to high fare outliers
plt.show()

# Visualization 6: Family Size Impact
plt.figure(figsize=(10, 6))
sns.countplot(x='FamilySize', hue='Survived', data=titanic)
plt.title('Survival by Family Size')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = titanic.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot of numerical features
sns.pairplot(titanic[['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Survived']], 
             hue='Survived')
plt.suptitle('Pairplot of Numerical Features by Survival', y=1.02)
plt.show()