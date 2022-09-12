from sklearn.datasets import load_digits 
from sklearn.manifold import TSNE 
import seaborn as sns 
from matplotlib import pyplot as plt 


# 필요한 데이터를 로드합니다. 여기서는 0부터 9까지의 숫자 데이터 입니다. 
data = load_digits() 


# 설명을 위한 참고 부분 
# 로딩한 데이터의 첫번째 샘플을 보면 아래와 같습니다. 0은 하얀색이고 높은 숫자일 수록 검은 색에 가까움을 나타냅니다. 
# 0이 아닌 숫자들을 연결해보면 중앙부분에 하얀색(0)이 있는 숫자 0을 나타내고 있음을 알 수 있습니다. 

# >>> data.data[0] 

# [ 0., 0., 5., 13., 9., 1., 0., 0.,  

# 0., 0., 13., 15., 10., 15., 5., 0., 

# 0., 3., 15., 2., 0., 11., 8., 0., 

# 0., 4., 12., 0., 0., 8., 8., 0., 

# 0., 5., 8., 0., 0., 9., 8., 0., 

# 0., 4., 11., 0., 1., 12., 7., 0., 

# 0., 2., 14., 5., 10., 12., 0., 0., 

# 0., 0., 6., 13., 10., 0., 0., 0. ] 

# 실제로 타겟의 첫번째에는 첫번째 샘플의 정답인 0이 들어잇습니다. 

# >>> data.target[0] 

# 0 


# 축소한 차원의 수를 정합니다. 

n_components = 2 

# TSNE 모델의 인스턴스를 만듭니다. 
model = TSNE(n_components=n_components) 

# data를 가지고 TSNE 모델을 훈련(적용) 합니다. 
X_embedded = model.fit_transform(data.data) 

# 훈련된(차원 축소된) 데이터의 첫번째 값을 출력해 봅니다.  

print(X_embedded[0]) 

# [65.49378 -7.3817754] 


# 차원 축소된 데이터를 그래프로 만들어서 화면에 출력해 봅니다. 

palette = sns.color_palette("bright", 10) 
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=data.target, legend='full', palette=palette) 

plt.show()