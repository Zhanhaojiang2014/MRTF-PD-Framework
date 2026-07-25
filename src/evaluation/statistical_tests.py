import numpy as np

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon
)


from sklearn.utils import resample




def normality_test(
    values
):

    stat,p = shapiro(
        values
    )


    return {

        "statistic":stat,

        "p_value":p

    }





def paired_test(
    mrtf,
    baseline
):


    normal = (
        shapiro(
            mrtf-baseline
        )[1] > 0.05
    )


    if normal:

        stat,p = ttest_rel(
            mrtf,
            baseline
        )

        test="paired_t_test"


    else:

        stat,p = wilcoxon(
            mrtf,
            baseline
        )

        test="wilcoxon"



    return {

        "test":
        test,

        "statistic":
        stat,

        "p_value":
        p

    }




def bootstrap_ci(
    values,
    iterations=1000,
    confidence=0.95
):


    means=[]


    for _ in range(
        iterations
    ):

        sample=resample(
            values
        )

        means.append(
            np.mean(sample)
        )



    lower=np.percentile(
        means,
        ((1-confidence)/2)*100
    )


    upper=np.percentile(
        means,
        (confidence+(1-confidence)/2)*100
    )


    return lower,upper





def cohens_d(
    x,
    y
):


    diff=np.mean(x)-np.mean(y)


    pooled=np.sqrt(
        (
            np.var(x)
            +
            np.var(y)
        )/2
    )


    return diff/pooled





def cliffs_delta(
    x,
    y
):

    greater=0
    lower=0


    for i in x:

        for j in y:

            if i>j:
                greater+=1

            elif i<j:
                lower+=1



    return (
        greater-lower
    )/(len(x)*len(y))
