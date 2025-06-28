# 📊 Documentation des Datasets OULAD

## 🎯 Vue d'ensemble

Le projet utilise le dataset **OULAD (Open University Learning Analytics Dataset)** qui contient des données sur l'apprentissage en ligne d'étudiants universitaires. L'objectif est d'identifier les facteurs d'abandon scolaire grâce au data mining.

## 📁 Structure des fichiers

### 1. **studentInfo.csv** (32,593 lignes × 12 colonnes)
**Informations démographiques et académiques des étudiants**

| Colonne | Type | Description | Valeurs possibles |
|---------|------|-------------|-------------------|
| `code_module` | object | Code du module | AAA, BBB, CCC, DDD, EEE, FFF, GGG |
| `code_presentation` | object | Session du module | 2013J, 2014J, 2013B, 2014B |
| `id_student` | int64 | Identifiant unique de l'étudiant | 3,733 à 2,716,795 |
| `gender` | object | Sexe | M, F |
| `region` | object | Région géographique | 13 régions (East Anglian, Scotland, etc.) |
| `highest_education` | object | Niveau d'éducation le plus élevé | HE Qualification, A Level, Lower Than A Level, Post Graduate, No Formal quals |
| `imd_band` | object | Indice de privation multiple | 90-100%, 80-90%, ..., 0-10% (3.41% manquants) |
| `age_band` | object | Tranche d'âge | 0-35, 35-55, 55<= |
| `num_of_prev_attempts` | int64 | Nombre de tentatives précédentes | 0 à 6 |
| `studied_credits` | int64 | Crédits étudiés | 30 à 655 |
| `disability` | object | Handicap | Y, N |
| `final_result` | object | Résultat final | Pass, Withdrawn, Fail, Distinction |

### 2. **courses.csv** (22 lignes × 3 colonnes)
**Informations sur les cours et modules**

| Colonne | Type | Description |
|---------|------|-------------|
| `code_module` | object | Code du module |
| `code_presentation` | object | Session du module |
| `module_presentation_length` | int64 | Durée du module en jours (234-269) |

### 3. **assessments.csv** (206 lignes × 6 colonnes)
**Définition des évaluations**

| Colonne | Type | Description | Valeurs possibles |
|---------|------|-------------|-------------------|
| `code_module` | object | Code du module | 7 modules |
| `code_presentation` | object | Session du module | 4 sessions |
| `id_assessment` | int64 | Identifiant de l'évaluation | 1,752 à 40,088 |
| `assessment_type` | object | Type d'évaluation | TMA, Exam, CMA |
| `date` | float64 | Date de l'évaluation | 12 à 261 (5.34% manquants) |
| `weight` | float64 | Poids de l'évaluation | 0 à 100 |

### 4. **studentAssessment.csv** (173,912 lignes × 5 colonnes)
**Résultats des étudiants aux évaluations**

| Colonne | Type | Description |
|---------|------|-------------|
| `id_assessment` | int64 | Identifiant de l'évaluation |
| `id_student` | int64 | Identifiant de l'étudiant |
| `date_submitted` | int64 | Date de soumission (-11 à 608) |
| `is_banked` | int64 | Si le résultat est "banké" | 0, 1 |
| `score` | float64 | Score obtenu | 0 à 100 (0.10% manquants) |

### 5. **studentRegistration.csv** (32,593 lignes × 5 colonnes)
**Inscriptions et désinscriptions des étudiants**

| Colonne | Type | Description |
|---------|------|-------------|
| `code_module` | object | Code du module |
| `code_presentation` | object | Session du module |
| `id_student` | int64 | Identifiant de l'étudiant |
| `date_registration` | float64 | Date d'inscription (-322 à 167, 0.14% manquants) |
| `date_unregistration` | float64 | Date de désinscription (-365 à 444, 69.10% manquants) |

### 6. **vle.csv** (6,364 lignes × 6 colonnes)
**Environnement d'apprentissage virtuel (VLE)**

| Colonne | Type | Description |
|---------|------|-------------|
| `id_site` | int64 | Identifiant du site VLE |
| `code_module` | object | Code du module |
| `code_presentation` | object | Session du module |
| `activity_type` | object | Type d'activité | 20 types (resource, oucontent, url, etc.) |
| `week_from` | float64 | Semaine de début | 0 à 29 (82.39% manquants) |
| `week_to` | float64 | Semaine de fin | 0 à 29 (82.39% manquants) |

### 7. **studentVle.csv** (10,655,280 lignes × 6 colonnes)
**Interactions des étudiants avec le VLE**

| Colonne | Type | Description |
|---------|------|-------------|
| `code_module` | object | Code du module |
| `code_presentation` | object | Session du module |
| `id_student` | int64 | Identifiant de l'étudiant |
| `id_site` | int64 | Identifiant du site VLE |
| `date` | int64 | Date de l'interaction (-25 à 269) |
| `sum_click` | int64 | Nombre de clics | 1 à 6,977 |

## 🔗 Relations entre les tables

```
studentInfo ←→ studentRegistration (id_student, code_module, code_presentation)
studentInfo ←→ studentAssessment (id_student)
studentInfo ←→ studentVle (id_student, code_module, code_presentation)

courses ←→ studentInfo (code_module, code_presentation)
courses ←→ assessments (code_module, code_presentation)
courses ←→ vle (code_module, code_presentation)

assessments ←→ studentAssessment (id_assessment)
vle ←→ studentVle (id_site, code_module, code_presentation)
```

## 📊 Statistiques clés

- **32,593 étudiants** uniques
- **7 modules** différents (AAA à GGG)
- **4 sessions** (2013J, 2014J, 2013B, 2014B)
- **173,912 résultats** d'évaluations
- **10,655,280 interactions** VLE
- **Taux d'abandon** : à calculer depuis `final_result = 'Withdrawn'`

## 🎯 Variables cibles pour la prédiction

### Variable principale : **Abandon scolaire**
- Créée à partir de `studentInfo.final_result`
- `abandon = 1` si `final_result = 'Withdrawn'`
- `abandon = 0` sinon

### Variables prédictives potentielles :
1. **Démographiques** : sexe, région, âge, niveau d'éducation, handicap
2. **Académiques** : tentatives précédentes, crédits étudiés, scores aux évaluations
3. **Comportementales** : interactions VLE, dates d'inscription/désinscription
4. **Contextuelles** : module, session, durée du module

## 🚀 Prochaines étapes

1. **Exploration univariée** : Distribution des variables
2. **Analyse bivariée** : Relations avec l'abandon
3. **Feature engineering** : Création de nouvelles variables
4. **Modélisation** : Classification binaire (abandon vs non-abandon)
5. **Évaluation** : Métriques de performance et interprétabilité

## 📝 Notes importantes

- **Valeurs manquantes** : Principalement dans `imd_band` (3.41%) et `date_unregistration` (69.10%)
- **Dates négatives** : Les dates négatives indiquent des événements avant le début officiel du cours
- **Échelle des données** : Le dataset couvre plusieurs années et modules
- **Qualité des données** : Bonne qualité générale avec peu de valeurs aberrantes 