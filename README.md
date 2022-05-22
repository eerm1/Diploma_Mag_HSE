# Контентные рекомендательные системы на основе NLP моделей с вниманием 

Список датасетов: 
* Netflix Prize Dataset - состоит из примерно 100,000,000 рейтингов для 17,770 фильмов, оцененых 480,189 пользователями. Каждая оценка в датасете для обучения имеет следующий вид: пользователь, фильм, оценка. Пользователи и фильмы представлены как целые идентефикаторы, а рейтинги - оценки от 1 to 5.

* Anime Recommendation Dataset - содержит информацию о различных оценках 73516 пользователей на 12294 аниме. Пользователи и аниме представлены как целые идентефикаторы, а рейтинги - оценки от 1 to 10.

* MovieLens Dataset - GroupLens Research собрали информацию об оценках пользователей фильмам с сайта MovieLens. Датасет содержит 25000000 оценок для 62000 фильмов, которые были оценены 162000 пользователями. Пользователи и фильмы представлены как целые идентефикаторы, а рейтинги - оценки от 1 to 5.

**Оценка моделей**

Все метрики считались при k = 1,5,10,15,20. 

* **Anime Recommendation Dataset**

| Model  | ndcg_at_k | mnap_at_k | recall_at_k |
| ------------- | ------------- | ------------- | ------------- |
| ALS |  **0.097816, 0.100579, 0.103228, 0.106354, 0.109209** | **0.097816, 0.069687, 0.059904, 0.055404, 0.052965** | **0.097816, 0.107920, 0.116659, 0.124860, 0.131985** |
| SVD | **0.062626, 0.064192, 0.065461, 0.066330, 0.067461** | **0.001876, 0.004828, 0.006733, 0.008031, 	0.009056** | **0.062626, 0.064327, 0.064755, 0.063718, 0.062498** |
| SVDpp | **0.122519, 0.118179, 0.116480, 0.116406, 0.116521** | **0.003157, 0.007956, 0.010994,0.013185, 0.014985** | **0.122519, 0.116259,0.113068, 	0.111119,	0.108635** |
| LightFM | Content Cell  | Content Cell  | Content Cell  |
| BPR | **0.373276, 0.307713, 0.285390, 0.272327, 0.262579**  | **0.373276, 0.232278, 0.185644, 0.162787,	0.147870**  | **0.373276,	0.311045,	0.296084,	0.064947, 0.069216** |
| EMF | Content Cell  | Content Cell  | Content Cell  |

* **Netflix Prize Dataset**

| Model  | ndcg_at_k | mnap_at_k | recall_at_k |
| ------------- | ------------- | ------------- | ------------- |
| ALS |  **0.097816, 0.100579, 0.103228, 0.106354, 0.109209** | **0.097816, 0.069687, 0.059904, 0.055404, 0.052965** | **0.097816, 0.107920, 0.116659, 0.124860, 0.131985** |
| SVD | **0.045000, 0.048177, 0.050426, 0.062545, 0.069978** | **0.002045, 0.005076, 0.007072, 0.009857, 0.011935** | **0.045000, 0.047468, 0.050443, 0.124860, 0.131985** |
| SVDpp |  **0.050200, 0.052975, 0.055069, 0.067251, 0.074632** | **0.002147, 0.005234, 0.007281, 	0.010087, 0.012230** | **0.050200,	0.052178, 0.054946, 0.069866, 0.074400** |
| LightFM | Content Cell  | Content Cell  | Content Cell  |
| BPR | **0.555085, 0.483373, 0.449662, 0.428659, 0.412252** | **0.555085, 0.393037, 0.330420, 	0.295096, 0.271345** | **0.555085,	0.466102, 0.426271, 0.402825, 0.384746** |
| EMF | **0.79729, 0.883013,  0.921076, 0.93172, 0.935212** | **0.09358, 0.04928, 0.03581, 0.02995, 0.02652** | **0.00374, 0.01881, 0.0373, 0.05576, 0.07408**  | 

* **MovieLens Dataset**

| Model  | ndcg_at_k | mnap_at_k | recall_at_k |
| ------------- | ------------- | ------------- | ------------- |
| ALS |  **0.097816, 0.100579, 0.103228, 0.106354, 0.109209** | **0.097816, 0.069687, 0.059904, 0.055404, 0.052965** | **0.097816, 0.107920, 0.116659, 0.124860, 0.131985** |
| SVD |  **0.111347,0.111347,0.100176, 0.098163, 	0.098088** | **0.004628, 0.010045, 0.013967, 0.016738, 0.018901** | **0.111347, 0.098621,	0.095228, 0.089077,	0.084889** |
| SVDpp | **0.118770,0.103273,0.102237, 0.104446, 	0.105692** | **0.003764, 0.008817,	0.013112, 0.016495, 0.019342** | **0.118770,0.100742,	0.098303, 0.098339,	0.095228** |
| LightFM | Content Cell  | Content Cell  | Content Cell  |
| BPR | **0.542373, 0.467349,0.441273, 0.420986, 0.402739** | **0.542373, 0.376836, 0.323525, 0.290566, 0.265794** | **0.542373, 0.449153, 0.420763, 0.397458, 0.376059** |
| EMF | **0.87526, 0.91824, 0.93915, 0.94904, 0.95434** | **0.01, 0.05623, 0.07867, 0.0893, 0.10131** | **0.03024, 0.03024, 0.03024, 0.03024, 0.03024** |

