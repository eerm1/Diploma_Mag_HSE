# Контентные рекомендательные системы на основе NLP моделей с вниманием 

Все метрики считались при k = 1,5,10,15,20. 

Оценка моделей: 


* Anime Recommendation Dataset 

| Model  | ndcg_at_k | mnap_at_k | recall_at_k |
| ------------- | ------------- | ------------- | ------------- |
| ALS |  **0.097816, 0.100579, 0.103228, 0.106354, 0.109209** | **0.097816, 0.069687, 0.059904, 0.055404, 0.052965** | **0.097816, 0.107920, 0.116659, 0.124860, 0.131985** |
| SVD | **0.045000, 0.048177, 0.050426, 0.062545, 0.069978** | **0.002045, 0.005076, 0.007072, 0.009857, 0.011935** | **0.045000, 0.047468, 0.050443, 0.124860, 0.131985** |
| SVDpp | Content Cell  | Content Cell  | Content Cell  |
| LightFM | Content Cell  | Content Cell  | Content Cell  |
| BPR | **0.373276, 0.307713, 0.285390, 0.272327, 0.262579**  | **0.373276, 0.232278, 0.185644, 0.162787,	0.147870**  | **0.373276,	0.311045,	0.296084,	0.064947, 0.069216** |
| EMF | Content Cell  | Content Cell  | Content Cell  |

* Netflix Prize Dataset 

| Model  | ndcg_at_k | mnap_at_k | recall_at_k |
| ------------- | ------------- | ------------- | ------------- |
| ALS |  **0.097816, 0.100579, 0.103228, 0.106354, 0.109209** | **0.097816, 0.069687, 0.059904, 0.055404, 0.052965** | **0.097816, 0.107920, 0.116659, 0.124860, 0.131985** |
| SVD | **0.045000, 0.048177, 0.050426, 0.062545, 0.069978** | **0.002045, 0.005076, 0.007072, 0.009857, 0.011935** | **0.045000, 0.047468, 0.050443, 0.124860, 0.131985** |
| SVDpp | Content Cell  | Content Cell  | Content Cell  |
| LightFM | Content Cell  | Content Cell  | Content Cell  |
| BPR | Content Cell  | Content Cell  | Content Cell  |
| EMF | Content Cell  | Content Cell  | Content Cell  |

* MovieLens Dataset 

| Model  | ndcg_at_k | mnap_at_k | recall_at_k |
| ------------- | ------------- | ------------- | ------------- |
| ALS |  **0.097816, 0.100579, 0.103228, 0.106354, 0.109209** | **0.097816, 0.069687, 0.059904, 0.055404, 0.052965** | **0.097816, 0.107920, 0.116659, 0.124860, 0.131985** |
| SVD | Content Cell  | Content Cell  | Content Cell  |
| SVDpp | Content Cell  | Content Cell  | Content Cell  |
| LightFM | Content Cell  | Content Cell  | Content Cell  |
| BPR | Content Cell  | Content Cell  | Content Cell  |
| EMF | Content Cell  | Content Cell  | Content Cell  |
