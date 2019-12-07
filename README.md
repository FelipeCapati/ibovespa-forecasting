# Modelo hibrido de previsão do IBOVESPA baseado em ARIMA e Deep Learning #

### Abstract ###

###### TODO: ESCREVER AQUI O ABSTRACT EM INGLES ######

Keywords — ARIMA, Deep Learning, IBOVESPA.

## I.	 INTRODUÇÃO ##
Esse projeto tem como objetivo modelar o comportamento do indice IBOVESPA utilizando um sistema híbrido baseado em
ARIMA e Deep Learning. O ARIMA é uma técnica estatística para modelagem de sistemas univariaveis que não possuem
tendencia e que tenham uma variação constante, resumidamente é uma técnica capaz de desenhar o comportamento linear
da função de estudo.<br> 
O residuo proveniente do ARIMA descreve o comportamento não linear do sistema juntamente com o erro do sistema.<br>
A utilização de uma Rede Neural Profunda para modelagem do residuo do ARIMA vem especificadamente pelo fato de que
essas redes são capazes de criar modelos de multivariáveis com resultados não lineares correspondentes ao residuo
do ARIMA.

## II.	FUNDAMENTAÇÃO TEÓRICA ##

### A.	LINEAR DISCRIMINANT ANALYSIS ###


## III.	METODOLOGIA ##
Para o projeto vigente foi utilizado python juntamente com o Notebook Jupyter para prototipar o modelo do sistema.
A base do sistema foi feita utilizando o Statsmodels e o Keras com Tensorflow.<br>
Tendo como base a fundamentação teórica abordada em II, o modelo esta proposto em <b>"./hybrid_model.ipynb"</b>.<br>

Para o teste e análise do sistema utilizou-se o dataset fornecido pelo Yahoo Finance do indice IBOVESPA da data 27/11/2017 até
26/11/2019 correspondendo aos ultimos dois anos desde o inicio do desenvolvimento desse projeto.<br>
O dataset é composto pelos seguintes dados:<br>

* Abertuda<br>
* Fechamento<br>
* Máxima<br>
* Mínima<br>
* Aj. Fechamento<br>
* Volume<br>

![Alt text](images/ibv-show-graph-01.png?)<br>

![Alt text](images/ibv-show-graph-02.png?)<br>

Tendo em vista que o modelo ARIMA é univariável foi feita uma análise de correlação dos dados do dataset e foi escolhido
o preço de fechamento do dia para seguir com a modelagem.

![Alt text](images/ibv-corr-graph-01.png?)<br>

Para fins metodológicos foi feito o test de Dickey-Fuller para saber o parametro d e analisado o gráfico de autocorrelação
e de autocorrelação parcial para saber, respectivamente, o parametro q e p.

Uma das bibliotecas do Python utilizada no projeto contem um metodo nomeado de "auto_arima" que minimiza o AIC e o BIC do
ARIMA proposto para descobrir os melhores parametros do modelo. Em IV será discutido com detalhes a utilização e resultado
do mesmo.

Como parametros de entrada da rede neural foi utilizado os mesmo dados utilizados pelo ARIMA com o lag correspondente do
modelo proposto pelo "auto_arima" e o fittedvalue do ARIMA do dia analisado para modelar o residuo do mesmo dia.

## IV. RESULTADOS ##
Os detalhes das implementações dos problemas propostos na metodologia pode ser analisados em <b>"./LDA.ipynb"</b> 
ou <b>"LDA.html"</b>.<br>
O Primeiro experimento foi feito utilizando o Iris Dataset, na qual temos a utilização do LDA para um espaço bidimensional
plotado nos gráficos a seguir.<br>
O primeiro gráfico é o resultado da redução da dimensionalidade proposta pelo LDA na qual busca maximizar a distância
entre as classes e minimizar a distância dos dados dentro da mesma classe.

![Alt text](images/ex1-graph01.png?)

O segundo gráfico plota os valores das classes em um espaço bidimensional, sem nenhum tipo de segmentação.

![Alt text](images/ex1-graph02.png?)

O terceiro gráfico plota os valores sepados em três classes utilizando o k-mean.

![Alt text](images/ex1-graph03.png?)

O segundo experimento foi feito pré-processando os dados de entrada utilizando PCA com uma, duas e três componentes
principais para, posteriormente, utilizar o LDA.

A seguir tem-se os gráficos dos experimentos seguindo a mesma ordem no primeiro experimentos.<br><br>
Utilização de uma componente principal.

![Alt text](images/ex2-pc1-graph01.png?)<br>
![Alt text](images/ex2-pc1-graph02.png?)

Utilização de duas componente principal.

![Alt text](images/ex2-pc2-graph01.png?)<br>
![Alt text](images/ex2-pc2-graph02.png?)<br>
![Alt text](images/ex2-pc2-graph03.png?)

Utilização de três componente principal.

![Alt text](images/ex2-pc3-graph01.png?)<br>
![Alt text](images/ex2-pc3-graph02.png?)<br>
![Alt text](images/ex2-pc3-graph03.png?)


## V. CONCLUSÃO ##
Dado os resultados vistos em IV podemos inferir que o LDA chegou ao resultado esperado e graficamente é possível inferir
que o erro de classificação é relativamente baixo.<br>
Como não foi utilizado nenhum indicador de performance dos métodos abordados, podemos avaliar a difenreça entre os
experimentos apenas de forma visual.<br>

Um ponto muito nítido na avaliação é que as Virginicas são muito próximas das Versicolor e que dependendo dos valores
de sépala e de pétala elas podem se confundir, porém as Setosas diferenciam-se bem dentro dos dados propostos.<br>

O dataset proposto tem dados de pétala e de sépala e seu comprimento é proporcional a largura, ou seja, são variáveis
altamente correlacionadas. Quando utiliza-se o PCA com uma componente principal, pode-se ver claramente que as classes
de Virginicas e Versicolor se confundem mais, provavelmente, devido ao fato de existirem duas variaveis com informações
relevantes para o modelo. Porém a não utilização do PCA, utilização de duas ou três componentes principais são muito
semelhante entre si, relembrando que essa inferencia é puramente qualitativa devido a não utilização de indicadores de
performance para os modelos abordados.

## VI. AGRADECIMENTOS ##

Agradecimentos especiais a CAPES e ao Centro Universitário FEI por financiar o mestrado que está em curso; 
ao professor Reinaldo Bianchi por proporcionar visões sobre o mundo acadêmico e orientar trabalhos científicos 
com o objetivo de lapidar os conhecimentos abordados em sala; aos meus pais e a minha família que sempre me 
apoiaram em meio a dificuldades.

## VII. REFERÊNCIAS ##

[1] S. Raschka, Linear Discriminant Analysis, 08/2014, link: https://sebastianraschka.com/Articles/2014_python_lda.html, acessado em 11/2019<br>
[2] S. Balakrishnama, A. Ganapathiraju, LINEAR DISCRIMINANT ANALYSIS - A BRIEF TUTORIAL, Institute for Signal and Information Processing, Department of Electrical and Computer Engineering.
[3]	R. Bianchi, Tópicos Especiais em Aprendizagem, 2019, ppt slide Centro Universitário FEI.

https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
