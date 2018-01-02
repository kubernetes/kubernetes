set -e;

# maintain a cached copy of coreos pxe image

if [ -z "${IMG_URL}" -o -z "${ITMP}" -o -z "${V}" ]; then
	exit 1
fi

if [ ${V} -eq 3 ]; then
	set -x
fi

# coreos gpg signing key
# pub   4096R/93D2DCB4 2013-09-06
# uid       [ unknown] CoreOS Buildbot (Offical Builds) <buildbot@coreos.com>
# sub   4096R/74E7E361 2013-09-06 [expired: 2014-09-06]
# sub   4096R/E5676EFC 2014-09-08 [expired: 2015-09-08]
# sub   4096R/1CB5FA26 2015-08-31 [expires: 2017-08-30]
# sub   4096R/B58844F1 2015-11-20 [revoked: 2016-05-16]
# sub   4096R/2E16137F 2016-05-16 [expires: 2017-05-16]
GPG_LONG_ID="50E0885593D2DCB4"
GPG_KEY="-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: GnuPG v2

mQINBFIqVhQBEADjC7oxg5N9Xqmqqrac70EHITgjEXZfGm7Q50fuQlqDoeNWY+sN
szpw//dWz8lxvPAqUlTSeR+dl7nwdpG2yJSBY6pXnXFF9sdHoFAUI0uy1Pp6VU9b
/9uMzZo+BBaIfojwHCa91JcX3FwLly5sPmNAjgiTeYoFmeb7vmV9ZMjoda1B8k4e
8E0oVPgdDqCguBEP80NuosAONTib3fZ8ERmRw4HIwc9xjFDzyPpvyc25liyPKr57
UDoDbO/DwhrrKGZP11JZHUn4mIAO7pniZYj/IC47aXEEuZNn95zACGMYqfn8A9+K
mHIHwr4ifS+k8UmQ2ly+HX+NfKJLTIUBcQY+7w6C5CHrVBImVHzHTYLvKWGH3pmB
zn8cCTgwW7mJ8bzQezt1MozCB1CYKv/SelvxisIQqyxqYB9q41g9x3hkePDRlh1s
5ycvN0axEpSgxg10bLJdkhE+CfYkuANAyjQzAksFRa1ZlMQ5I+VVpXEECTVpLyLt
QQH87vtZS5xFaHUQnArXtZFu1WC0gZvMkNkJofv3GowNfanZb8iNtNFE8r1+GjL7
a9NhaD8She0z2xQ4eZm8+Mtpz9ap/F7RLa9YgnJth5bDwLlAe30lg+7WIZHilR09
UBHapoYlLB3B6RF51wWVneIlnTpMIJeP9vOGFBUqZ+W1j3O3uoLij1FUuwARAQAB
tDZDb3JlT1MgQnVpbGRib3QgKE9mZmljYWwgQnVpbGRzKSA8YnVpbGRib3RAY29y
ZW9zLmNvbT6JAjkEEwECACMFAlIqVhQCGwMHCwkIBwMCAQYVCAIJCgsEFgIDAQIe
AQIXgAAKCRBQ4IhVk9LctFkGD/46/I3S392oQQs81pUOMbPulCitA7/ehYPuVlgy
mv6+SEZOtafEJuI9uiTzlAVremZfalyL20RBtU10ANJfejp14rOpMadlRqz0DCvc
Wuuhhn9FEQE59Yk3LQ7DBLLbeJwUvEAtEEXq8xVXWh4OWgDiP5/3oALkJ4Lb3sFx
KwMy2JjkImr1XgMY7M2UVIomiSFD7v0H5Xjxaow/R6twttESyoO7TSI6eVyVgkWk
GjOSVK5MZOZlux7hW+uSbyUGPoYrfF6TKM9+UvBqxWzz9GBG44AjcViuOn9eH/kF
NoOAwzLcL0wjKs9lN1G4mhYALgzQx/2ZH5XO0IbfAx5Z0ZOgXk25gJajLTiqtOkM
E6u691Dx4c87kST2g7Cp3JMCC+cqG37xilbV4u03PD0izNBt/FLaTeddNpPJyttz
gYqeoSv2xCYC8AM9N73Yp1nT1G1rnCpe5Jct8Mwq7j8rQWIBArt3lt6mYFNjuNpg
om+rZstK8Ut1c8vOhSwz7Qza+3YaaNjLwaxe52RZ5svt6sCfIVO2sKHf3iO3aLzZ
5KrCLZ/8tJtVxlhxRh0TqJVqFvOneP7TxkZs9DkU5uq5lHc9FWObPfbW5lhrU36K
Pf5pn0XomaWqge+GCBCgF369ibWbUAyGPqYj5wr/jwmG6nedMiqcOwpeBljpDF1i
d9zMN4kCHAQQAQIABgUCUipXUQAKCRDAr7X91+bcxwvZD/0T4mVRyAp8+EhCta6f
Qnoiqc49oHhnKsoN7wDg45NRlQP84rH1knn4/nSpUzrB29bhY8OgAiXXMHVcS+Uk
hUsF0sHNlnunbY0GEuIziqnrjEisb1cdIGyfsWUPc/4+inzu31J1n3iQyxdOOkrA
ddd0iQxPtyEjwevAfptGUeAGvtFXP374XsEo2fbd+xHMdV1YkMImLGx0guOK8tgp
+ht7cyHkfsyymrCV/WGaTdGMwtoJOxNZyaS6l0ccneW4UhORda2wwD0mOHHk2EHG
dJuEN4SRSoXQ0zjXvFr/u3k7Qww11xU0V4c6ZPl0Rd/ziqbiDImlyODCx6KUlmJb
k4l77XhHezWD0l3ZwodCV0xSgkOKLkudtgHPOBgHnJSL0vy7Ts6UzM/QLX5GR7uj
do7P/v0FrhXB+bMKvB/fMVHsKQNqPepigfrJ4+dZki7qtpx0iXFOfazYUB4CeMHC
0gGIiBjQxKorzzcc5DVaVaGmmkYoBpxZeUsAD3YNFr6AVm3AGGZO4JahEOsul2FF
V6B0BiSwhg1SnZzBjkCcTCPURFm82aYsFuwWwqwizObZZNDC/DcFuuAuuEaarhO9
BGzShpdbM3Phb4tjKKEJ9Sps6FBC2Cf/1pmPyOWZToMXex5ZKB0XHGCI0DFlB4Tn
in95D/b2+nYGUehmneuAmgde87kCDQRSKlZGARAAuMYYnu48l3AvE8ZpTN6uXSt2
RrXnOr9oEah6hw1fn9KYKVJi0ZGJHzQOeAHHO/3BKYPFZNoUoNOU6VR/KAn7gon1
wkUwk9Tn0AXVIQ7wMFJNLvcinoTkLBT5tqcAz5MvAoI9sivAM0Rm2BgeujdHjRS+
UQKq/EZtpnodeQKE8+pwe3zdf6A9FZY2pnBs0PxKJ0NZ1rZeAW9w+2WdbyrkWxUv
jYWMSzTUkWK6533PVi7RcdRmWrDMNVR/X1PfqqAIzQkQ8oGcXtRpYjFL30Z/LhKe
c9Awfm57rkZk2EMduIB/Y5VYqnOsmKgUghXjOo6JOcanQZ4sHAyQrB2Yd6UgdAfz
qa7AWNIAljSGy6/CfJAoVIgl1revG7GCsRD5Dr/+BLyauwZ/YtTH9mGDtg6hy/So
zzDAM8+79Y8VMBUtj64GQBgg2+0MVZYNsZCN209X+EGpGUmAGEFQLGLHwFoNlwwL
1Uj+/5NTAhp2MQA/XRDTVx1nm8MZZXUOu6NTCUXtUmgTQuQEsKCosQzBuT/G+8Ia
R5jBVZ38/NJgLw+YcRPNVo2S2XSh7liw+Sl1sdjEW1nWQHotDAzd2MFG++KVbxwb
cXbDgJOB0+N0c362WQ7bzxpJZoaYGhNOVjVjNY8YkcOiDl0DqkCk45obz4hG2T08
x0OoXN7Oby0FclbUkVsAEQEAAYkERAQYAQIADwUCUipWRgIbAgUJAeEzgAIpCRBQ
4IhVk9LctMFdIAQZAQIABgUCUipWRgAKCRClQeyydOfjYdY6D/4+PmhaiyasTHqh
iui2DwDVdhwxdikQEl+KQQHtk7aqgbUAxgU1D4rbLxzXyhTbmql7D30nl+oZg0Be
yl67Xo6X/wHsP44651aTbwxVT9nzhOp6OEW5z/qxJaX1B9EBsYtjGO87N854xC6a
QEaGZPbNauRpcYEadkppSumBo5ujmRWc4S+H1VjQW4vGSCm9m4X7a7L7/063HJza
SYaHybbu/udWW8ymzuUf/UARH4141bGnZOtIa9vIGtFl2oWJ/ViyJew9vwdMqiI6
Y86ISQcGV/lL/iThNJBn+pots0CqdsoLvEZQGF3ZozWJVCKnnn/kC8NNyd7Wst9C
+p7ZzN3BTz+74Te5Vde3prQPFG4ClSzwJZ/U15boIMBPtNd7pRYum2padTK9oHp1
l5dI/cELluj5JXT58hs5RAn4xD5XRNb4ahtnc/wdqtle0Kr5O0qNGQ0+U6ALdy/f
IVpSXihfsiy45+nPgGpfnRVmjQvIWQelI25+cvqxX1dr827ksUj4h6af/Bm9JvPG
KKRhORXPe+OQM6y/ubJOpYPEq9fZxdClekjA9IXhojNA8C6QKy2Kan873XDE0H4K
Y2OMTqQ1/n1A6g3qWCWph/sPdEMCsfnybDPcdPZp3psTQ8uX/vGLz0AAORapVCbp
iFHbF3TduuvnKaBWXKjrr5tNY/njrU4zEADTzhgbtGW75HSGgN3wtsiieMdfbH/P
f7wcC2FlbaQmevXjWI5tyx2m3ejG9gqnjRSyN5DWPq0m5AfKCY+4Glfjf01l7wR2
5oOvwL9lTtyrFE68t3pylUtIdzDz3EG0LalVYpEDyTIygzrriRsdXC+Na1KXdr5E
GC0BZeG4QNS6XAsNS0/4SgT9ceA5DkgBCln58HRXabc25Tyfm2RiLQ70apWdEuoQ
TBoiWoMDeDmGLlquA5J2rBZh2XNThmpKU7PJ+2g3NQQubDeUjGEa6hvDwZ3vni6V
vVqsviCYJLcMHoHgJGtTTUoRO5Q6terCpRADMhQ014HYugZVBRdbbVGPo3YetrzU
/BuhvvROvb5dhWVi7zBUw2hUgQ0g0OpJB2TaJizXA+jIQ/x2HiO4QSUihp4JZJrL
5G4P8dv7c7/BOqdj19VXV974RAnqDNSpuAsnmObVDO3Oy0eKj1J1eSIp5ZOA9Q3d
bHinx13rh5nMVbn3FxIemTYEbUFUbqa0eB3GRFoDz4iBGR4NqwIboP317S27NLDY
J8L6KmXTyNh8/Cm2l7wKlkwi3ItBGoAT+j3cOG988+3slgM9vXMaQRRQv9O1aTs1
ZAai+Jq7AGjGh4ZkuG0cDZ2DuBy22XsUNboxQeHbQTsAPzQfvi+fQByUi6TzxiW0
BeiJ6tEeDHDzdLkCDQRUDREaARAA+Wuzp1ANTtPGooSq4W4fVUz+mlEpDV4fzK6n
HQ35qGVJgXEJVKxXy206jNHx3lro7BGcJtIXeRb+Wp1eGUghrG1+V/mKFxE4wulN
tFXoTOJ//AOYkPq9FG12VGeLZDckAR4zMhDwdcwsJ208hZzBSslJOWAuZTPoWple
+xie4B8jZiUcjf10XaWvBnlx4EPohhvtv5VEczZWNvGa/0VDe/FfI4qGknJM3+d0
kvXK/7yaFpdGwnY3nE/V4xbwx2tggqQRXoFmYbjogGHpTcdXkWbGEz5F7mLNwzZ/
voyTiZeukZP5I45CCLgiB+g2WTl8cm3gcxrnt/aZAJCAl/eclFeYQ/Xiq8sK1+U2
nDEYLWRygoZACULmLPbUEVmQBOw/HAufE98sb36MHcFss634h2ijIp9/wvnX9GOE
LgX4hgqkgM85QaMeaS3d2+jlMu8BdsMYxPkTumsEUShcFtAYgtrNrPSayHtV6I9I
41ISg8EIr9qEhH1xLGvSA+dfUvXqwa0cIBxhI3bXOa25vPHbT+SLtfQlvUvKySIb
c6fobw2Wf1ZtM8lgFL3f/dHbT6fsvK6Jd/8iVMAZkAYFbJcivjS9/ugXbMznz5Wv
g9O7hbQtXUvRjvh8+AzlASYidqSd6neW6o+i2xduUBlrbCfW6R0bPLX+7w9iqMaT
0wEQs3MAEQEAAYkERAQYAQIADwUCVA0RGgIbAgUJAeEzgAIpCRBQ4IhVk9LctMFd
IAQZAQIABgUCVA0RGgAKCRClqWY15Wdu/JYcD/95hNCztDFlwzYi2p9vfaMbnWcR
qzqavj21muB9vE/ybb9CQrcXd84y7oNq2zU7jOSAbT3aGloQDP9+N0YFkQoYGMRs
CPiTdnF7/mJCgAnXei6SO+H6PIw9qgC4wDV0UhCiNh+CrsICFFbK+O+Jbgj+CEN8
XtVhZz3UXbH/YWg/AV/XGWL1BT4bFilUdF6b2nJAtORYQFIUKwOtCAlI/ytBo34n
M6lrMdMhHv4MoBHP91+Y9+t4D/80ytOgH6lq0+fznY8Tty+ODh4WNkfXwXq+0TfZ
fJiZLvkoXGD+l/I+HE3gXn4MBwahQQZl8gzI9daEGqPF8KYX0xyyKGo+8yJG5/WG
lfdGeKmz8rGP/Ugyo6tt8DTSSqJv6otAF/AWV1Wu/DCniehtfHYrp2EHZUlpvGRl
7Ea9D9tv9BKYm6S4+2yD5KkPu4qp3r6glVbePPCLeZ4NLQCEIpKakIERfxk66JqZ
Tb5XI9HKKbnhKunOoGiL5SMXVsS67Sxt//Ta/3vSaLC3wnVwN5OeXNaa04Yx7jg/
wtMJ9Jz0EYFtVv2NLizEeGCI8iPJOyMWOy+twCIk5zmvwsLu5MKmg1tLI2mtCTYz
qo8uVIqETlojxIqAhRYtmeiYKf2fZs5um3+Sjv28v4nw3VfQgibTKc2uBjeqxxOe
XGw0ysKnS2VO72SK879+EADd3HoF9U80odCgN5T6aljhaNaruqmG4CvBdRyzp3EQ
9RP7jPOEhcM00etw572orviK9AqCk+zwvfzEFbt/uC7zOpO0BJ8fnMAZ0Zn/fF8s
88zR4zq6BBq9WD4RCmazw2G6IyGXHvVAWi8UxoNjNoJJosLyLauFdPPUeoye5PxE
g+fQew3behcCaebjZwUA+xZMj7dfwcNXlDa4VkCDHzTfU43znawBo9avB8hNwMeW
CZYINmym+LSKyQnz3sirTpYcjorxtov1fyml8413tDJoOvkotSX9o3QQgbBPsyQ7
nwLTscYc5eklGRH7iytXOPI+29EPpfRHX2DAnVyTeVSFPEr79tIsijy02ZBZTiKY
lBlJy/Cj2C5cGhVeQ6v4jnj1Nt3sjHkZlVfmipSYVfcBoID1/4r2zHl4OFlLCjvk
XUhbqhm9xWV8NdmItO3BBSlIEksFunykzz1HM6shvzw77sM5+TEtSsxoOxxys+9N
ItCl8L6yf84A5333pLaUWh5HON1J+jGGbKnUzXKBsDxGSvgDcFlyVloBRQShUkv3
FMem+FWqt7aA3/YFCPgyLp7818VhfM70bqIxLi0/BJHp6ltGN5EH+q7Ewz210VAB
ju5IO7bjgCqTFeR3YYUN87l8ofdARx3shApXS6TkVcwaTv5eqzdFO9fZeRqHj4L9
PrkCDQRV5KHhARAAz9Qk17qaFi2iOlRgA4WXhn5zkr9ed1F1HGIJmFB4J8NIVkTZ
dt2UfRBWw0ykOB8m1sWLEfimP2FN5urnfsndtc1wEVrcuc7YAMbfUgxbTc/o+gTy
dpVCKmGrL10mZeOmioFQuVT9s1qzIII/gDbiSLRVDb75F6/aag7mDsJFGtUqStpN
mR0AHyrLOY/jYVLlTr8dAfX2Z2aBifpJ/nPaw29FkTBCQvyC84+cReTT3RiUOXQ3
EL4zLaYm/VTtLlAnZ4IYADpGijFHw2c4jcBWZ/72Wb6TUk9lg2b6M6THfCwNieJB
CwCf6VHyKBebbYZYHiuZB5GILfdm4aSclRACVXT3seTZQh8yeCYLMYyieceeHesO
M/4rC5iLujbNsVN+95z0SuRMPlpd3mfExFYeeH6SO/EgTL5cCXwP6L2R2vP67gSs
P01HBTOAOzEzXQQ4IY1kK2zUjbJJBx8HylvcYLlbsRce1uvMmCR/b7QWJEXR/7VX
qjCtmYIwroxhGiMpH5Fssh0z62BiBXDLc0iSKVBD3P36Uv++o51aDOg/V928ve/D
4ISf28IiNnVIg1/zrUy2+LpFSUkU+Szjd77leUSjOTFnpyHQhlsZuG02S4SO1opX
O6HblhuEjCEcw2TUDgvXb9hsuj+C+d4DFdTdQ/bPZ0sc2351wkiqn4JhMekAEQEA
AYkERAQYAQIADwUCVeSh4QIbAgUJA8JnAAIpCRBQ4IhVk9LctMFdIAQZAQIABgUC
VeSh4QAKCRAH+p7THLX6JlrhD/9W+hAjebjCRuNfcAoMFVujrSNgiR7o6aH5Re0q
cPITQ4ev4muNEl+L1AMcBiAr7Ke7fdEhhSdWiBOutlig3VFRRaX6kOQlS5h+lazi
JQc84VR9iBnWMsfK3WadMYmRkTR4P/lHsGTvczD8Qhl7kha8BGbm1a4SgWuF3FOR
xEWkimz8AIpaozf+vD4CV2rVSaJ0oHRLJXqQHrhWuBy73NVF4wa/7lxDi7Q3PA8p
6Rr5Kr+IVuPVUvxJOVLEUfGgpEnMnTbRu322HvUqeLNrNnSCdJKePuoy2Sky0K+/
82O877nFysagTeO4tbLr+OiVG/6ORiInn1y7uQjwLgrz8ojDjGMNmqnNW8ACYhey
4ko3L9xdep0VhxaBwjVWBU6fhbogSVkCRhjz8h2sLGdItLzDxp69y0ncf931H0e5
DAB7VbURuKh6P8ToQQhWUD5zIOCyxFXMQPA63pxd7mQooCpaWK1i80J/fRA5TBIP
Lqty2NEP3aTePelrBdqiQol/aPQ3ugtrnP/PLLlJ0zxg/YNGgBFRwNHgnu7HxOOr
E4gap8prvZCKC/05A71AXwj6u2h9so9jSrE5slrOgfh9v9w9AyuQzNMG/2l1Cli4
UpeVqy07Qn27evjEbad6HT1vmrPJE3A/D9hzEFPWMM+sPOWH+4L2Qekoy954M5fW
CQ2aoL3+EACDFKJIEp/Xc8n3CRuqxxNwRij6EJ2jYZZURQONwtumFXDD0LKF7Upc
ZrOiG4i2qojp0WQWarQuITmiyds0jtDg+xhdQUZ3HgjhN/MNT3O0klTXsZ4AYrys
9yDhdC030kD/CqKxTOJJCz8z2of2xXY9/rKpTvZAra+UBEzNKb7F+dQ3kclZF6CG
MnNY51KBXi1xRAv9J8LdsdNsTOhoZG/2s4vbVCkgKWF60NRh/jw7JFM9YYre8+qM
R1bbaW/uW4Ts9XopaG5+auS9mYFDgICdyXqrwzUo4PLbnTqTxni6Ldt525wye+/h
ex5ssLi+PMhCalcWEAKUYYW/CfDyZqwtRDoBAKwStcV5DrcK28YBzheMAEcGI7dE
xVHYpET+49ERwTvYQtwKqZSDBoivrQg5MdJpu8Ncj126DbN2lwQQpIsMmq93jOCv
DEPTdTUOs5XzLv8YTYDKiyxm3IKPsSvElnoI/wedO4EscldAAQqNKo/6pzI+K4Eh
ifyLT1GOMN7PCaHzW449DrSJNd3yL7xkzNtrphw32a9qLJ43sWFrF21EjG1IQgUV
4XOz01Q2Hp4H1l1YE11MbSL/+TarNTbEfhzv6tS3eNrlU/MQDLsUn76c4hi2tAbK
X8FjXVJ/8MWi91Z0pHcLzhYZYn2IACvaaUh06HyyAIiDlgWRC7zgMbkCDQRWT38I
ARAAzWz3KxYiRJ04sltTwnndeFYaBMJySA+wN2Y2Re5/sS1C97+ryNfGcj50MQ7m
RbSXzqvfvlbvgiLjSL337UwahrXboLcYxbmVzsIG/aXiCogPlJ3ooyd6Krn/p4CO
tzhVDlReBSkNdwUxusAsAVdSDpJVk/JOTil49g7jx3angVqHmI/oPyPIcGhNJlBV
ofVxJZKVWSsmP8rsWYZ0LHNdSngt7uhYb8BO57sSfKpT0YJpP7i5/Au3ZXohBa9K
tEJELX/WJe95i38ysq/xedRwKg7Zt9aNND7Tiic+3DRONvus3StvN6dHEhM84RNW
bk/XDmjjCk92cB6Gm32HPDk8rnAfXug/rJFWD/CzGwCvxmPuikXEZesHLCdrgzZh
VGQ9BcAh8oxz1QcPQXr7TCk8+cikSemQrVmqJPq2rvdVpZIzF91ZCpAfT28e0y/a
DxbrfS83Ytk+90dQOR8rStGNVnrwT/LeMn1ytV7oK8e2sIj1HFUYENQxy5jVjR3Q
tcTbVoOYLvZ83/wanc4GaZnxZ7cJguuKFdqCR5kq4b7acjeQ8a76hrYI57Z+5JDs
L+aOgGfCqCDx2IL/bRiwY1pNDfTCPhSSC054yydG3g6pUGk9Kpfj+oA8XrasvR+d
D4d7a2cUZRKXU29817isfLNjqZMiJ/7LA11I6DeQgPaRK+kAEQEAAYkCHwQoAQgA
CQUCVzocNwIdAgAKCRBQ4IhVk9LctGVfEADBBSjZq858OE932M9FUyt5fsYQ1p/O
6zoHlCyGyyDdXNu2aDGvhjUVBd3RbjHW87FiiwggubZ/GidCSUmv/et26MAzqthl
5CJgi0yvb5p2KeiJvbTPZEN+WVitAlEsmN5FuUzD2Q7BlBhFunwaN39A27f1r3av
qfy6AoFsTIiYHVP85HscCaDYc2SpZNAJYV4ZcascuLye2UkUm3fSSaYLCjtlVg0m
Wkcjp7rZFQxqlQqSjVGarozxOYgI+HgKaqYF9+zJsh+26kmyHRdQY+Pznpt+PXjt
EQVsdzh5pqr4w4J8CnYTJKQQO4T08cfo13pfFzgqBGo4ftXOkLLDS3ZgFHgx00fg
70MGYYAgNME7BJog+pO5vthwfhQO6pMT08axC8sAWD0wia362VDNG5Kg4TQHFARu
Ao51e+NvxF8cGi0g1zBEfGMCFwlAlQOYcI9bpk1xx+Z8P3Y8dnpRdg8VK2ZRNsf/
CggNXrgjQ2cEOrEsda5lG/NXbNqdDiygBHc1wgnoidABOHMT483WKMw3GBao3JLF
L0njULRguJgTuyI9ie8HLH/vfYWXq7t5o5sYM+bxAiJDDX+F/dp+gbomXjDE/wJ/
jFOz/7Cp9WoLYttpWFpWPl4UTDvfyPzn9kKT/57OC7OMFZH2a3LxwEfaGTgDOvA5
QbxS5txqnkpPcokERAQYAQgADwUCVk9/CAIbAgUJAeEzgAIpCRBQ4IhVk9LctMFd
IAQZAQgABgUCVk9/CAAKCRCGM/sTtYhE8RLLD/0bK5unOEb1RsuzCqL7IWPr+Z6i
7smZ0tmrTF58a3St64DjR3WYuv/RnhYyh8xCtBod7ZoIl2S+Azavevx22KWXPQgR
twhlCJFsnDoG9C5Kj0BqUrtyk+9nlGeIMOUPjMJJocEaB9yHZs7J9KFNyqpEY7x2
XW6HTDihsBdaOUu814g6C4gLiXydwbQMzU2Crefc1w/fWhSxjqiyUlKp571jeauW
uUdtbQmwk/Kvq9yreHkEWN4MHs2HuBwwBmbj0KDFFDA2u6oUvGlRTfwomTiryXDr
1tOgiySucdFVrx+6zPBMcqlXqsVDsx8sr+u7PzIsHO9NT+P3wYQpmWhwKCjLX5KN
6Xv3d0aAr7OYEacrED1sqndIfXjM5EcouLFtw/YESA7Px8iRggFVFDN0GY3hfoPJ
gHpiJj2KYyuVvNe8dXpsjOdPpFbhTPI1CoA12woT4vGtfxcI9u/uc7m5rQDJI+FC
R9OtUYvtDUqtE/XYjqPXzkbgtRy+zwjpTTdxn48OaizVU3JOW+OQwW4q/4Wk6T6n
zNTpQDHUmIdxsAAbZjBJwkE4Qkgtl8iUjS0hUX05ixLUwn0ZuGjeLcK9O/rqynPD
qd9gdeKo5fTJ91RhJxoBSFcrj21tPOa0PhE/2Zza24AVZIX5+AweD9pie8QIkZLM
k6yrvRFqs2YrHUrc5emkD/4lGsZpfSAKWCdc+iE5pL434yMlp73rhi+40mbCiXMO
gavdWPZSDcVe+7fYENx0tqUyGZj2qKluOBtxTeovrsFVllF9fxzixBthKddA6IcD
QdTb076t/Ez51jX1z/GRPzn8yWkDEvi3L9mfKtfuD4BRzjaVw8TtNzuFuwz2PQDD
BtFXqYMklA67cdjvYdffO7MeyKlNjKAutXOr/Or70rKkk2wZLYtSeJIDRwUSsPdK
ncbGLEKvfoBKOcOmjfZKjnYpIDDNqAsMrJLIwyo+6NSUtq84Gba6QjPYLvJ9g4P2
99dIYzFxu/0Zy4q9QgfjJOav3GUQT1fRhqqRS11ffXFqClJKqsKSChcPhNhK5wt6
Ab6PVbd9RQhI8ImLQ81PWn708rOr1dQTQfPvJrHBBrEolUw/0y7SxPmQZUkYlXiT
6bvsUa2n2f4ZzIgtYtZ5JSuoqcut/jmeUQE1TUUyG+9HVMfmhlhjNO0pDiFdSAgj
k+DyTd5lUVz3tPGFliIDq7O/sgDq6xtSlGKvQt/gRoYstrillyxfIVqR10C2t2kC
BXKSX3uQmbx3OaX8JtZ2uMjmKZb2iovfSf8qLSu49qrsNS9Etqvda0EXqaHeX+K8
NjENoQEdXZUnBRJg9VVa0HkPiFSFIwF8IPWewm0DocZil66bp/wrHVsJkw7AwE/z
JrkCDQRXOi4eARAA+cAKfT0IoViuCxqa6uPteVC8/qp8ZiEPri0neCt+khngPpCX
9JseOyRJEzwt9+31XgzsCWlfW5BWrLBd3F4caRqucu3ZnE68Qtrw6kcOsJ8LSiok
/uu1XnXW1mgpRxlu0i83YVM6+BrIXroP22SWVxkDkAXDlgvFmIvrh9TG43uSRjmg
riSnJ7EOgDXDrZ5mTlnlGHb6EGpHJHoJsfp3JdBAh4oNGBBHf5fZZhBiUIJSGwbL
g8oEzOuycNor9mEiJPaAyPm22braWRgvX7beOca60eNGIuQSZ8ML3G6rog/pNdbN
gLf1hvrfl7NJCJJ0iB7BPYw8e5+xPEHNLrJI6NjFCbD0dlHnuq79ePc9bPQALa/6
lIICOCAZJYDCf7S2dHqkHCOnr8F2A2qwAqP5IlVqdS7sSy7D9wDDYis7jlMw8vVW
jqcL6MNxJDk3h/0ns7Ad5TNfJnLUnUbYWeH5QYbPsGgqQomhSWBvhCZkILnE7Rpb
tjl55/CvTXN1L6jyi9qJeSoWORjwhTlACKDzlsLRTO24sM/KjKDajYrqU3CRVDQG
gQL0yU3qDz/mql+awQAMUS9ckaf/ohBM8SrCandNvE/+as426Mf6/FH6R7kntJpp
YQZJMwq0XlyueadWs8xrCjrXnXFijvrVkaZhlCfJRZPEdI76hGscRp8Sr6kAEQEA
AYkERAQYAQgADwUCVzouHgIbAgUJAeEzgAIpCRBQ4IhVk9LctMFdIAQZAQgABgUC
VzouHgAKCRBI+blqLhYTf6o8D/0WqjCOqB4rAv29MGpz5SZbk57TbQrKfjneSDVe
CsvgofUBL6z9yA2jEanIh76Lo6r5ZnvF8I4pDImiRCjhZ+4vDOKaO5yvrNKruusr
+ZA6DDPwjlhnRPqW8Sm1YGl1VqAqQEjib4I7dbGb5qpR/PkAj64UDtLtbMfx6Zb9
B9ZJvYEiWUbAEQWUohRhw6vT/qS07GrKgG35JFiJKrNPSFEh/YOLKq+vLVZwDKX9
1Tvabs3MuNFIavuMiGaoqv4/JVRA1Iw3E9zCsXgFhIfQll4XvrrPXiGAllFzaqX2
9PnvqMngjPRDTh+jHNUjFv8MNvhs1o3jc1pQAJT5JIpPQJJpbnNnrYoCJoBO0kfJ
04zEDznHkuVbLRn2pxWsCrF2Agwm4GB3YSenEW8AKcmtS4ov0Yaw5csY3fXUDXjB
aPR9dweNWT/kaY5V4NUwOutecnZ0o0yDc57GGIjFhTcULMdOCE6DbSTfljqcoAoP
IydzQ4rlMdmTkiM5k2F/jDHCURersqF8Naro7Nx2fKokPrLKUst+pFBBwbeTO9tW
EbOnl/ypHeRW9XA31sZ0yvvSwUrWnHC+UDpHPzvaAGleAOK7gGyJehVIw9BhgZB1
LplkbkGgpS8L/3CAcaQ488MP5NK0peO+ED/ocNhi1tC/cHbLXtDiz/eG/1rIdxkO
h3D61WiyD/42Oj2h4BHt5qTS12By95po4avzgqaV3PFYi9Rx6tBvzwnD7x2UeGk4
wzFdb2V4LWoe6bqMokxbUMWJgP5faWDT6/urhBt4GYcBxX0b3l9qBs20hP5JVHGX
208gOW5cjfHrTNiHiY4/CbQrbAdO24CUYZtYEmDNdHN+KHrlLLjkf0v5yGjVK2XB
qs8l6upA7xBGHAF7U/XkLYrvyusqqWdvdGHGHthbLBzjceO+4N+lb6RyHRuF6kgb
LdCcaKfCMUs/v1ZXgYGhdk7NWFHFDoF8DByHwluoihd10OudGPFg7ydTc6+V3kt9
SN1/iQbk2/rHffI1tm28MfBvN+K/Da+Y+EAqTbUDHl6O30mSGZjLl1xJxvWoezU9
8TdPCxy7L9XRFfqZlBJAo8cxRIPHpqKIaRy0wn616xCDfUSQ9NBLlDITL4d7tNvD
C9hLpehFKMKIEct5WDfaQIWQe2o1fjVsU2Is2wXVmdi9A7X3q7yWVA766zQTxQO6
1TcgyoJM9k2DxncsmwXIa8oD6KP4VYtrtsx8r4VXPEjHucCjPe+qgyY65wBPXSl5
U21AiUuGGegFQwRD6L7ZqT3K5JLDlK/kkaV3l8i0onfJ+5CytOB2T6QPQnJ4Ynch
K9w3EiyDrgzl0IpotQXxOBGHoCtcxUZvkNeOIxAb8QwkWnhgkljMyQ==
=nV4h
-----END PGP PUBLIC KEY BLOCK-----
"

# prints passed flags if verbosity level is lower; to be used inside
# output capture (backticks or $())
function be_quiet() {
	local verbosity	#verbosity level
	local flags	#silencing flags to use

	verbosity=$1
	flags=$2

	if [ $verbosity -lt 3 ]; then
		printf '%s\n' "${flags}"
	fi
}

# gpg verify a file using the provided key
function gpg_verify() {
	local file	#file to verify
	local sigfile	#signature file (assumed to be suffixed form of file to verify)
	local key	#signing key
	local keyid	#signing key signature
	local verbosity	#verbosity level

	local quiet
	local gpghome

	file=$1
	sigfile=$2
	key=$3
	keyid=$4
	verbosity=$5

	quiet=$(be_quiet $verbosity '--quiet --no-verbose')
	gpghome=$(mktemp -d)
	trap "{ rm -rf '${gpghome}'; }" RETURN EXIT
	if ! gpg --homedir="${gpghome}" --batch --quiet --import <<<"${key}"; then
		return 1
	fi
	if ! gpg --homedir="${gpghome}" --batch ${quiet} --trusted-key "${keyid}" --verify "${sigfile}" "${file}"; then
		return 1
	fi
	return 0
}

function do_wget() {
	local out	#output file
	local url	#url of a file to be downloaded
	local verbosity	#verbosity level

	local quiet
	local short_out

	out=$1
	url=$2
	verbosity=$3

	quiet=$(be_quiet $verbosity '--quiet')
	if [ "${quiet}" ]; then
		# strip the working directory from output path, so we get
		# something like build-rkt/tmp/coreos-common/pxe.img
		# instead of /home/foo/projects/coreos/rkt/build-rkt/...
		short_out="${out#${PWD}/}"
		printf '  %-12s %s\n' 'WGET' "${url} => ${short_out}"
	fi
	wget ${quiet} --tries=20 --output-document="${out}" "${url}" # the wget default for retries is 20 times.
}

function cat_to_stderr_if_verbose() {
	local file
	local verbosity

	file=$1
	verbosity=$2

	if [ -z $(be_quiet $verbosity 'empty-if-verbose') ]; then
		cat "${file}" >&2
	fi
}

# maintain an gpg-verified url cache, assumes signature available @ $url.sig
function cache_url() {
	local cache	#verified cache, will be downloaded from the url if bad or missing
	local url	#url of the file to be downloaded
	local key	#key used for verification
	local keyid	#id of a key used for verification
	local verbosity	#verbosity level

	local urlhash
	local sigfile
	local sigurl
	local gpgout

	cache=$1
	url=$2
	key=$3
	keyid=$4
	verbosity=$5

	urlhash=$(echo -n "${url}" | md5sum)
	sigfile="${cache}.${urlhash%% *}.sig"
	sigurl="${url}.sig"

	gpgout=$(mktemp)
	trap "{ rm -f '${gpgout}'; }" RETURN EXIT
	# verify the cached copy if it exists
	if ! gpg_verify "${cache}" "${sigfile}" "${key}" "${keyid}" "${verbosity}" 2>"${gpgout}"; then
		# refresh the cache on failure, and verify it again
		cat_to_stderr_if_verbose "${gpgout}" "${verbosity}"
		do_wget "${cache}" "${url}" "${verbosity}"
		do_wget "${sigfile}" "${sigurl}" "${verbosity}"
		if ! gpg_verify "${cache}" "${sigfile}" "${key}" "${keyid}" "${verbosity}" 2>"${gpgout}"; then
			# print an error if verification failed
			cat "${gpgout}" >&2
			return 1
		fi
		cat_to_stderr_if_verbose "${gpgout}" "${verbosity}"
	else
		cat_to_stderr_if_verbose "${gpgout}" "${verbosity}"
	fi

	# file $cache exists and can be trusted
	touch "${cache}"
}

# cache pxe image
cache_url "${ITMP}/pxe.img" "${IMG_URL}" "${GPG_KEY}" "${GPG_LONG_ID}" "${V}"
