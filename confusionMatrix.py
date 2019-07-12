import numpy as np

# 1 para grávida, 0 para não grávida
valores_reais    = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
valores_preditos = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

def get_confusion_matrix(reais, preditos, labels):
    """
    Uma função que retorna a matriz de confusão para uma classificação binária
    
    Args:
        reais (list): lista de valores reais
        preditos (list): lista de valores preditos pelo modelos
        labels (list): lista de labels a serem avaliados.
            É importante que ela esteja presente, pois usaremos ela para entender
            quem é a classe positiva e quem é a classe negativa
    
    Returns:
        Um numpy.array, no formato:
            numpy.array([
                [ tp, fp ],
                [ fn, tn ]
            ])
    """
    # não implementado
    if len(labels) > 2:
        return None

    if len(reais) != len(preditos):
        return None
    
    # considerando a primeira classe como a positiva, e a segunda a negativa
    
    negative_class = labels[0]
    true_class = labels[1]

    # valores preditos corretamente
    tp = 0
    tn = 0
    
    # valores preditos incorretamente
    fp = 0
    fn = 0
    
    for (indice, v_real) in enumerate(reais):
        v_predito = preditos[indice]

        # se trata de um valor real da classe positiva
        if v_real == negative_class:
            tp += 1 if v_predito == v_real else 0
            fp += 1 if v_predito != v_real else 0
        else:
            tn += 1 if v_predito == v_real else 0
            fn += 1 if v_predito != v_real else 0
    
    return np.array([
        # valores da classe positiva
        [ tp, fp ],
        # valores da classe negativa
        [ fn, tn ]
    ])

#get_confusion_matrix(reais=valores_reais, preditos=valores_preditos, labels=[1,0])
# array([[3, 1], [2, 4]])

#precisao = vp/(vp+fp)

matrizConfusao = get_confusion_matrix(reais=valores_reais, preditos=valores_preditos, labels=[1,0])

vp = matrizConfusao.item(0)
fp = matrizConfusao.item(1)
fn = matrizConfusao.item(2)
vn = matrizConfusao.item(3)
print(vn)

precisao = vp / (vp + fp)
sensibilidade = vp / (vp + fn)
acuracia = (vp + vn) / (vp + vn + fp + fn)
especificidade = vn / (vn + fp)
erroGradiente = (fp + fn) / (vp + vn + fp + fn)

print("precisao = ", precisao)
print("sensibilidade = ", sensibilidade)
print("acuracia = ", acuracia)
print("especificidade = ",especificidade)
print("erro = ", erroGradiente)
