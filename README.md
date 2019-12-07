# Modelo hibrido de previsão do IBOVESPA baseado em ARIMA e Deep Learning #

### Abstract ###

###### TODO: ESCREVER AQUI O ABSTRACT EM INGLES ######

Keywords — ARIMA, Deep Learning, IBOVESPA.

## I.	 INTRODUÇÃO ##
Esse projeto tem como objetivo modelar o comportamento do indice IBOVESPA utilizando um sistema híbrido baseado em
ARIMA e Deep Learning. O ARIMA é uma técnica estatística para modelagem de sistemas univariaveis que não possuem
tendencia e que tenham uma variação constante, resumidamente é uma técnica capaz de desenhar o comportamento linear
da função de estudo.<br> 
Pode-se modelar o indice como uma composição de um sinal linear e um sinal não-linear. O ARIMA (ferramenta utilizada
para análise de séries temporais) é responsável por fazer a análise da parte linear do modelo e, modelos computacionais/
matemáticos/estatísticos, tais como como SVM (Suport Vector Machine) e ANN (Atificial Neural Network) são responsáveis
pela predição do residuo do ARIMA formada pela parte não linear do dado e o erro do sistema. Sistemas já desenvolvidos
em projetos da área, tais como [1] [2] [3] [4] [5] [6] utilizados como inspiração para o desenvolvimento de um modelo baseado em Deep
Learning para análise do resíduo do ARIMA.<br>
A utilização de uma Rede Neural Profunda para modelagem do residuo do ARIMA vem especificadamente pelo fato de que
essas redes são capazes de criar modelos de multivariáveis com resultados não lineares correspondentes ao residuo
do ARIMA.

## II.	FUNDAMENTAÇÃO TEÓRICA ##

### A.	ARIMA ###

ARIMA é uma ferramenta estatística focada em séries temporais, ou seja, uma série de observações que acontecem baseadas
em eventos temporais iguais. Existem dois tipos de séries temporais: séries que contém espectro contínuo e séries com 
espectro discreto [7].<br>
Modelos ARIMA são modelos auto-regressivos integrados de médias móveis representado da seguinte forma: ARIMA(p, d, q),
onde p é a ordem da parte AR (Auto Regressiva), q é a ordem da parte MA (Média Móvel) e d é o número de diferenças realizadas
nos dados até que a série se tornasse estacionária (caso a série seja estacionária d=0) (MORETIN, TOLOI, 2004, apud, [8]).<br>
Com base em [8] temos um fluxo base para a construção do modelo ARIMA:<br>

![Alt text](images/ibv-arima-flux-01.png?)<br>

* Seleção: é dada por qual modelo será utilizado.
* Estimação da ordem/ Estimação dos coeficientes: Qual os valores dados aos atributos p, d e q.
* Diagnóstico: Verificação se o modelo é adequado, ou seja, representa bem a série temporal.

Segundo (MORETIN, TOLOI, 2004, apud, [8]) a entrada do modelo ARIMA é supostamente um ruido branco e, para ser considerado
como tal deve conter as seguintes características:

* Conter Média 0;
* Conter Variancia constante;
* Conter Covariancia Nula

### A.I - Modelo AR e ARI ###
O modelo AR(p) é um modelo composto apenas pela parte auto-regressiva do ARIMA(p, 0, 0) assumindo a forma: [8]

![Alt text](images/ibv-arima-equation-01.png?)<br>

Em que:

![Alt text](images/ibv-arima-equation-02.png?)<br>

O modelo ARI(p, d) assume a parte integrativa do modelo, podendo-se reduzir a um modelo ARIMA(p, d, 0) assumindo a forma: [8]

![Alt text](images/ibv-arima-equation-03.png?)<br>

Em que:

![Alt text](images/ibv-arima-equation-04.png?)<br>

### A.II - Modelo MA ###
O modelo MA(q) é composto apenas pela parte das médias móveis (Moving Average), podendo-se reduzir a um modelo ARIMA(0, 0, q)
assumindo a forma: [8]

![Alt text](images/ibv-arima-equation-05.png?)<br>

Em que:

![Alt text](images/ibv-arima-equation-06.png?)<br>

### A.	Deep Learning ###

Aprendizagem Profunda (AP) ou Deep Learning (DL) é uma subdivisão do Aprendizado de Máquina. Implementando neurônios
baseados em modelos matemáticos, o Deep Learning é modelado a partir de como seria o aprendizado de um humano, no que
diz respeito a sinapse e os neurônios envolvidos, utilizando não mais redes neurais simples, porém redes neurais profundas
com muito mais camadas, muito mais neurônios, com mais de um estado de transformação não linear e uma capacidade de 
acuracidade muito maior [9].
Atualmente é um assunto muito difundido pela capacidade de processar um alto volume de dados e de resultar em informações
muito precisas. O Deep Learning deu espaço para o avanço do processamento de imagens, áudio, linguagem natural, entre outras aplicações.
No projeto em questão o Deep Learning entra como uma ferramenta para avaliar o resíduo do ARIMA e propor uma melhoria com
base nas mesmas informações que o ARIMA tem, juntamente com o valor retornado pelo mesmo.

## III.	METODOLOGIA ##
O projeto vigente foi desenvolvido utilizado python juntamente com o Notebook Jupyter para prototipar o modelo do sistema.
A base foi feita utilizando o Statsmodels e o Keras com Tensorflow, tendo como base a fundamentação teórica
abordada em II<br>

Os teste e a análise do sistema foram executados utilizando o dataset fornecido pelo Yahoo Finance [10] do indice IBOVESPA
da data 27/11/2017 até 26/11/2019 correspondendo aos ultimos dois anos desde o inicio do desenvolvimento desse projeto.<br>
O dataset é composto pelos seguintes dados:<br>

* Abertuda<br>
* Fechamento<br>
* Máxima<br>
* Mínima<br>
* Aj. Fechamento<br>
* Volume<br>

A seguir tem-se o OHLC do IBOVESPA e o gráfico do volume de transação.

![Alt text](images/ibv-show-graph-01.png?)<br>

![Alt text](images/ibv-show-graph-02.png?)<br>

Tendo em vista que o modelo ARIMA é univariável foi feita uma análise de correlação dos dados do dataset e foi escolhido
o preço de fechamento do dia para seguir com a modelagem.

![Alt text](images/ibv-corr-graph-01.png?)<br>

![Alt text](images/ibv-show-graph-03.png?)<br>

Foi utilizado o test de Dickey-Fuller para saber o parametro "d", a análise do mesmo pode ser vista a seguir:<br>
Para d = 0, temos:<br>
* p-value: 0.759863
* Aceita a hipótese nula

Para d = 1, temos:<br>
* p-value: 5.199467e-30
* Rejeita a hipótese nula

Sabendo que para:
* p-value > 0.05: Aceita a hipótese nula (H0), os dados têm uma raiz unitária e não são estacionários.
* p-value <= 0.05: Rejeita a hipótese nula (H0), os dados não têm uma raiz unitária e são estacionários.
 
A seguir tem-se o gráfico do valor de fechamento utilizando d = 1.<br>
![Alt text](images/ibv-show-graph-04.png?)<br>

Sabendo o valor de "d" o próximo passo é analisar o gráfico de autocorrelação e de autocorrelação parcial para saber,
respectivamente, o parametro "q" e "p" do ARIMA.

![Alt text](images/ibv-show-graph-05.png?)<br>

Existem técnicas analíticas não abordadas nesse projeto para extração dos valores de "p" e "q", porém devido a capacidades
computacionais e a possibilidade da utilização de uma biblioteca Python chamada "pmdarima" que contem um metodo nomeado 
de "auto_arima" que possibilita achar o modelo ARIMA que minimiza o Critério de Informação de Akaike (AIC) e o Critério 
de Informação Bayesiano (BIC) do modelo proposto para descobrir os melhores parametros do modelo.<br>
Seguindo a minimização do AIC e do BIC proposto por "pmdarima" temos o seguinte modelo:<br>

![Alt text](images/ibv-arima-model-01.png?)<br>

Os parametros de entrada da rede neural foram os mesmo dados utilizados pelo ARIMA com o lag correspondente do
modelo proposto pelo "auto_arima" e o fittedvalue do ARIMA do dia analisado para modelar o residuo do mesmo dia.<br>
Foi plotado o gráfico de correlação dos dados de entrada do modelo DL:

![Alt text](images/ibv-dl-graph-01.png?)<br>

A seguir tem-se o gráfico de loss (MSE - Mean Square Error) do modelo baseado nas épocas análisadas.

![Alt text](images/ibv-dl-graph-02.png?)<br>

## IV. RESULTADOS ##
Analisando o modelo ARIMA(4,1,4) proposto temos o seguinte resultado quando utilizamos o modelo simples (sem a utilização
do DL para calculo do residuo):<br>

![Alt text](images/ibv-arima-graph-01.png?)<br>

* MAPE: 0.00973482216584 <br>
* ME:   0.24113477603335 <br>
* MAE:  858.991871351844 <br>
* MPE:  0.00010688570412 <br>
* RMSE: 1103.29299537375 <br>

Forecasting utilizando 2 períodos:

![Alt text](images/ibv-arima-graph-00.png?)<br>

Fazendo uma análise do resíduo temos:<br>

![Alt text](images/ibv-arima-graph-02.png?)<br>

![Alt text](images/ibv-arima-graph-03.png?)<br>

Após o treinamento da rede, temos o seguinte gráfico, respectivamente, referente ao resíduo do sistema e do erro quadrático
do modelo utilizando o modelo ARIMA(4,1,4) e o modelo ARIMA + DL:

![Alt text](images/ibv-dl-graph-03.png?)<br>

![Alt text](images/ibv-dl-graph-04.png?)<br>

Erro Quadrático do Modelo ARIMA(4,1,4)

* count: 4.900000e+02<br>
* mean: 1.222407e+06<br>
* std: 1.966396e+06<br>
* min: 5.181610e+00<br>
* 25%: 1.213062e+05<br>
* 50%: 4.789919e+05<br>
* 75%: 1.347107e+06<br>
* max: 1.312116e+07<br>

Erro Quadrático do Modelo ARIMA(4,1,4) + DL

* count: 4.900000e+02<br>
* mean: 1.029466e+06<br>
* std: 1.756038e+06<br>
* min: 5.255445e+01<br>
* 25%: 8.162935e+04<br>
* 50%: 4.493298e+05<br>
* 75%: 1.182404e+06<br>
* max: 1.471125e+07<br>

Avaliando a média e desvio padrão do erro quadrático temos uma redução de 15.78% da média e 10.69% do desvio padrão.
 
## V. CONCLUSÃO ##
Dado os resultados vistos em IV podemos inferir que o modelo proposto utilizando ARIMA + Deep Learning proporciona maior
precisão do que o ARIMA puro devido a redução da média e do desvio padrão do erro quadratico do modelo.<br>
Mesmo para o modelo ARIMA puro, temos uma aproximação boa do modelo real com um MAPE: 0.00973 (detalhados em IV) e um
bom forecasting para dois dias.<br>
Resumidamente tem-se a possibilidade de modalagem do IBOVESPA utilizando apenas ARIMA e a obtenção de bons resultados que
podem ser melhorados ainda mais quando combina-se o modelo com Deep Learning para predição do resíduo do ARIMA.

## VI. Proposta para Projetos Futuros
Analisando o gráfico de fechamento para d=1 tempos um valor médio proximo a zero com uma variação não constante, ou seja,
existe a possibilidade de implementar modelo de Heteroscedasticidade Condicional Auto-Regressiva (ARCH) com chance
de modelagem melhor que o ARIMA, porém tal ajuste proposto pelo ARCH pode ser compensado pelo DL ou não.<br>
Aqui fica uma sujestão de modelo que pode ser testado em trabalhos futuros. Um modelo baseado em ARCH + DL comparado com
um modelo ARIMA + DL.

## VI. AGRADECIMENTOS ##

Agradecimentos especiais a CAPES e ao Centro Universitário FEI por financiar o mestrado que está em curso; 
ao professor Reinaldo Bianchi e ao João Chang Junior por proporcionar visões sobre o mundo acadêmico e orientar trabalhos científicos 
com o objetivo de lapidar os conhecimentos abordados em sala; aos meus pais e a minha família que sempre me apoiaram em meio a dificuldades.

## VII. REFERÊNCIAS ##

[1] P. F. Pai, C. S. Lin; A hybrid ARIMA and support vector machines model in stock price forecasting, Department of Industrial Engineering and Technology Management, Da-Yeh University, 2004, Taiwan<br>
[2] H. Yan, Z. Zou; Application of Hybrid ARIMA and Neural Network Model to Water Quality Time Series Forecasting, School of Economics and Management, Beihang University, Beijiung, 2013, China<br>
[3] L. Xiong, Y. Lu; Hybrid ARIMA-BPNN Model for Time Series Prediction of the Chinese Stock Market, Shanghai University, Shanghai, China, 2017<br>
[4] S. Sreelekshmy, et al; Stock Price Prediction Using LSTM, RNN and CNN-Sliding Window Model, Centre for Computational Engineering and Networking (CEN), Amrita School of Engineering, Coimbatore, India, 2017<br>
[5] J. H Wang, J. Y. Leu; Stock Market Trend Prediction Using ARIMA-based Neural Networks, Department of Eletrical Engineering, National Taiwan Ocean University, Taiwan, 1996.<br>
[6] G. P. Zhang; Time series forecasting using a hybrid ARIMA and neural network model, Department of Management J. Mack Robinson College of Business, Georgia State University, Atlanta, USA, 2001.<br>
[7] Ky M Vu, The ARIMA and VARIMA time series : their modelings, analyses and applications, Ottawa: AuLac Technologies, 2007, cap 3.<br>
[8] W. L. S. Andrade; Estimação de Modelos ARIMA/ARIMAX e Aplicação em Inferência de Perdas de Propano, Universidade Federal do Rio Grande do Norte, Centro de Técnologia, Natal, RN, 2009.<br>
[9] A. BHARDWAJ, W. DI, J. WEI; Deep Learning Essentials: Your hands-on guide to the fundamentals of deep learning and neural network modeling, 2018 Ed. Packt Publishing Ltd, Cap. 1, p 7 - 20
[10] Yahoo Finance, https://finance.yahoo.com/quote/%5EBVSP/history?p=%5EBVSP, acessado em 11/2019<br>
