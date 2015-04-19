function unique(arqw) {
        var a = [], i, j
        outer: for (i = 0; i < arqw.length; i++) {
                for (j = 0; j < a.length; j++) {
                        if (a[j] == arqw[i]) {
                                continue outer
                        }
                }
                a[a.length] = arqw[i]
        }
        return a
}


function unique(arqw) {
        var crap = [], i, j
        outer: for (i = 0; i < arqw.length; i++) {
                for (j = 0; j < crap.length; j++) {
                        if (crap[j] == arqw[i]) {
                                continue outer
                        }
                }
                crap[crap.length] = arqw[i]
        }
        return crap
}
