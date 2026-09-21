import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join("data", "train.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['AgeGroup'] = df['Age'].apply(lambda e: 'Niño' if e<12 else ('Joven' if e<18 else ('Adulto' if e<60 else 'Adulto mayor')))
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', errorbar=None, palette='Set2')
plt.title('Tasa de Supervivencia por Clase y Género')
plt.savefig(os.path.join(OUTPUT_DIR, 'supervivencia_clase_genero.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='AgeGroup', y='Survived', hue='AgeGroup', order=['Niño', 'Joven', 'Adulto', 'Adulto mayor'], palette='mako', errorbar=None, legend=False)
plt.title('Tasa de Supervivencia por Grupo de Edad')
plt.savefig(os.path.join(OUTPUT_DIR, 'supervivencia_grupo_edad.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Survived', y='Fare', hue='Survived', palette='Set1', showfliers=False, legend=False)
plt.title('Distribución de Tarifas Pagadas según Supervivencia')
plt.xticks([0, 1], ['No Sobrevivió', 'Sobrevivió'])
plt.savefig(os.path.join(OUTPUT_DIR, 'tarifa_vs_supervivencia.png'), bbox_inches='tight')
plt.close()

print("Análisis completado exitosamente. TEAM ALE Y CHARBEL")