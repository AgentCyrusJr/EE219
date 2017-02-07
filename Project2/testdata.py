from sklearn.datasets import fetch_20newgroups

categories = ['comp.graphics']
graphics_train = fetch_20newsgroups(sucset = 'train', categories = categories, shuffle=True, random_state=42)