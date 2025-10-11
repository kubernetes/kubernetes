# Benchmarks

## How to reproduce

For a comparison run:

    make bench

followed by [benchcmp](http://code.google.com/p/go/source/browse/misc/benchcmp benchcmp) on the resulting files:

    $GOROOT/misc/benchcmp $GOPATH/src/github.com/gogo/protobuf/test/mixbench/marshal.txt $GOPATH/src/github.com/gogo/protobuf/test/mixbench/marshaler.txt
    $GOROOT/misc/benchcmp $GOPATH/src/github.com/gogo/protobuf/test/mixbench/unmarshal.txt $GOPATH/src/github.com/gogo/protobuf/test/mixbench/unmarshaler.txt

Benchmarks ran on Revision:  11c56be39364

June 2013

Processor 2,66 GHz Intel Core i7 

Memory 8 GB 1067 MHz DDR3 

## Marshaler

<table>
<tr><td>benchmark</td><td>old ns/op</td><td>new ns/op</td><td>delta</td></tr>
<tr><td>BenchmarkNidOptNativeProtoMarshal</td><td>2656</td><td>889</td><td>-66.53%</td></tr>
<tr><td>BenchmarkNinOptNativeProtoMarshal</td><td>2651</td><td>1015</td><td>-61.71%</td></tr>
<tr><td>BenchmarkNidRepNativeProtoMarshal</td><td>42661</td><td>12519</td><td>-70.65%</td></tr>
<tr><td>BenchmarkNinRepNativeProtoMarshal</td><td>42306</td><td>12354</td><td>-70.80%</td></tr>
<tr><td>BenchmarkNidRepPackedNativeProtoMarshal</td><td>34148</td><td>11902</td><td>-65.15%</td></tr>
<tr><td>BenchmarkNinRepPackedNativeProtoMarshal</td><td>33375</td><td>11969</td><td>-64.14%</td></tr>
<tr><td>BenchmarkNidOptStructProtoMarshal</td><td>7148</td><td>3727</td><td>-47.86%</td></tr>
<tr><td>BenchmarkNinOptStructProtoMarshal</td><td>6956</td><td>3481</td><td>-49.96%</td></tr>
<tr><td>BenchmarkNidRepStructProtoMarshal</td><td>46551</td><td>19492</td><td>-58.13%</td></tr>
<tr><td>BenchmarkNinRepStructProtoMarshal</td><td>46715</td><td>19043</td><td>-59.24%</td></tr>
<tr><td>BenchmarkNidEmbeddedStructProtoMarshal</td><td>5231</td><td>2050</td><td>-60.81%</td></tr>
<tr><td>BenchmarkNinEmbeddedStructProtoMarshal</td><td>4665</td><td>2000</td><td>-57.13%</td></tr>
<tr><td>BenchmarkNidNestedStructProtoMarshal</td><td>181106</td><td>103604</td><td>-42.79%</td></tr>
<tr><td>BenchmarkNinNestedStructProtoMarshal</td><td>182053</td><td>102069</td><td>-43.93%</td></tr>
<tr><td>BenchmarkNidOptCustomProtoMarshal</td><td>1209</td><td>310</td><td>-74.36%</td></tr>
<tr><td>BenchmarkNinOptCustomProtoMarshal</td><td>1435</td><td>277</td><td>-80.70%</td></tr>
<tr><td>BenchmarkNidRepCustomProtoMarshal</td><td>4126</td><td>763</td><td>-81.51%</td></tr>
<tr><td>BenchmarkNinRepCustomProtoMarshal</td><td>3972</td><td>769</td><td>-80.64%</td></tr>
<tr><td>BenchmarkNinOptNativeUnionProtoMarshal</td><td>973</td><td>303</td><td>-68.86%</td></tr>
<tr><td>BenchmarkNinOptStructUnionProtoMarshal</td><td>1536</td><td>521</td><td>-66.08%</td></tr>
<tr><td>BenchmarkNinEmbeddedStructUnionProtoMarshal</td><td>2327</td><td>884</td><td>-62.01%</td></tr>
<tr><td>BenchmarkNinNestedStructUnionProtoMarshal</td><td>2070</td><td>743</td><td>-64.11%</td></tr>
<tr><td>BenchmarkTreeProtoMarshal</td><td>1554</td><td>838</td><td>-46.07%</td></tr>
<tr><td>BenchmarkOrBranchProtoMarshal</td><td>3156</td><td>2012</td><td>-36.25%</td></tr>
<tr><td>BenchmarkAndBranchProtoMarshal</td><td>3183</td><td>1996</td><td>-37.29%</td></tr>
<tr><td>BenchmarkLeafProtoMarshal</td><td>965</td><td>606</td><td>-37.20%</td></tr>
<tr><td>BenchmarkDeepTreeProtoMarshal</td><td>2316</td><td>1283</td><td>-44.60%</td></tr>
<tr><td>BenchmarkADeepBranchProtoMarshal</td><td>2719</td><td>1492</td><td>-45.13%</td></tr>
<tr><td>BenchmarkAndDeepBranchProtoMarshal</td><td>4663</td><td>2922</td><td>-37.34%</td></tr>
<tr><td>BenchmarkDeepLeafProtoMarshal</td><td>1849</td><td>1016</td><td>-45.05%</td></tr>
<tr><td>BenchmarkNilProtoMarshal</td><td>439</td><td>76</td><td>-82.53%</td></tr>
<tr><td>BenchmarkNidOptEnumProtoMarshal</td><td>514</td><td>152</td><td>-70.43%</td></tr>
<tr><td>BenchmarkNinOptEnumProtoMarshal</td><td>550</td><td>158</td><td>-71.27%</td></tr>
<tr><td>BenchmarkNidRepEnumProtoMarshal</td><td>647</td><td>207</td><td>-68.01%</td></tr>
<tr><td>BenchmarkNinRepEnumProtoMarshal</td><td>662</td><td>213</td><td>-67.82%</td></tr>
<tr><td>BenchmarkTimerProtoMarshal</td><td>934</td><td>271</td><td>-70.99%</td></tr>
<tr><td>BenchmarkMyExtendableProtoMarshal</td><td>608</td><td>185</td><td>-69.57%</td></tr>
<tr><td>BenchmarkOtherExtenableProtoMarshal</td><td>1112</td><td>332</td><td>-70.14%</td></tr>
</table>

<table>
<tr><td>benchmark</td><td>old MB/s</td><td>new MB/s</td><td>speedup</td></tr>
<tr><td>BenchmarkNidOptNativeProtoMarshal</td><td>126.86</td><td>378.86</td><td>2.99x</td></tr>
<tr><td>BenchmarkNinOptNativeProtoMarshal</td><td>114.27</td><td>298.42</td><td>2.61x</td></tr>
<tr><td>BenchmarkNidRepNativeProtoMarshal</td><td>164.25</td><td>561.20</td><td>3.42x</td></tr>
<tr><td>BenchmarkNinRepNativeProtoMarshal</td><td>166.10</td><td>568.23</td><td>3.42x</td></tr>
<tr><td>BenchmarkNidRepPackedNativeProtoMarshal</td><td>99.10</td><td>283.97</td><td>2.87x</td></tr>
<tr><td>BenchmarkNinRepPackedNativeProtoMarshal</td><td>101.30</td><td>282.31</td><td>2.79x</td></tr>
<tr><td>BenchmarkNidOptStructProtoMarshal</td><td>176.83</td><td>339.07</td><td>1.92x</td></tr>
<tr><td>BenchmarkNinOptStructProtoMarshal</td><td>163.59</td><td>326.57</td><td>2.00x</td></tr>
<tr><td>BenchmarkNidRepStructProtoMarshal</td><td>178.84</td><td>427.49</td><td>2.39x</td></tr>
<tr><td>BenchmarkNinRepStructProtoMarshal</td><td>178.70</td><td>437.69</td><td>2.45x</td></tr>
<tr><td>BenchmarkNidEmbeddedStructProtoMarshal</td><td>124.24</td><td>317.56</td><td>2.56x</td></tr>
<tr><td>BenchmarkNinEmbeddedStructProtoMarshal</td><td>132.03</td><td>307.99</td><td>2.33x</td></tr>
<tr><td>BenchmarkNidNestedStructProtoMarshal</td><td>192.91</td><td>337.86</td><td>1.75x</td></tr>
<tr><td>BenchmarkNinNestedStructProtoMarshal</td><td>192.44</td><td>344.45</td><td>1.79x</td></tr>
<tr><td>BenchmarkNidOptCustomProtoMarshal</td><td>29.77</td><td>116.03</td><td>3.90x</td></tr>
<tr><td>BenchmarkNinOptCustomProtoMarshal</td><td>22.29</td><td>115.38</td><td>5.18x</td></tr>
<tr><td>BenchmarkNidRepCustomProtoMarshal</td><td>35.14</td><td>189.80</td><td>5.40x</td></tr>
<tr><td>BenchmarkNinRepCustomProtoMarshal</td><td>36.50</td><td>188.40</td><td>5.16x</td></tr>
<tr><td>BenchmarkNinOptNativeUnionProtoMarshal</td><td>32.87</td><td>105.39</td><td>3.21x</td></tr>
<tr><td>BenchmarkNinOptStructUnionProtoMarshal</td><td>66.40</td><td>195.76</td><td>2.95x</td></tr>
<tr><td>BenchmarkNinEmbeddedStructUnionProtoMarshal</td><td>93.24</td><td>245.26</td><td>2.63x</td></tr>
<tr><td>BenchmarkNinNestedStructUnionProtoMarshal</td><td>57.49</td><td>160.06</td><td>2.78x</td></tr>
<tr><td>BenchmarkTreeProtoMarshal</td><td>137.64</td><td>255.12</td><td>1.85x</td></tr>
<tr><td>BenchmarkOrBranchProtoMarshal</td><td>137.80</td><td>216.10</td><td>1.57x</td></tr>
<tr><td>BenchmarkAndBranchProtoMarshal</td><td>136.64</td><td>217.89</td><td>1.59x</td></tr>
<tr><td>BenchmarkLeafProtoMarshal</td><td>214.48</td><td>341.53</td><td>1.59x</td></tr>
<tr><td>BenchmarkDeepTreeProtoMarshal</td><td>95.85</td><td>173.03</td><td>1.81x</td></tr>
<tr><td>BenchmarkADeepBranchProtoMarshal</td><td>82.73</td><td>150.78</td><td>1.82x</td></tr>
<tr><td>BenchmarkAndDeepBranchProtoMarshal</td><td>96.72</td><td>153.98</td><td>1.59x</td></tr>
<tr><td>BenchmarkDeepLeafProtoMarshal</td><td>117.34</td><td>213.41</td><td>1.82x</td></tr>
<tr><td>BenchmarkNidOptEnumProtoMarshal</td><td>3.89</td><td>13.16</td><td>3.38x</td></tr>
<tr><td>BenchmarkNinOptEnumProtoMarshal</td><td>1.82</td><td>6.30</td><td>3.46x</td></tr>
<tr><td>BenchmarkNidRepEnumProtoMarshal</td><td>12.36</td><td>38.50</td><td>3.11x</td></tr>
<tr><td>BenchmarkNinRepEnumProtoMarshal</td><td>12.08</td><td>37.53</td><td>3.11x</td></tr>
<tr><td>BenchmarkTimerProtoMarshal</td><td>73.81</td><td>253.87</td><td>3.44x</td></tr>
<tr><td>BenchmarkMyExtendableProtoMarshal</td><td>13.15</td><td>43.08</td><td>3.28x</td></tr>
<tr><td>BenchmarkOtherExtenableProtoMarshal</td><td>24.28</td><td>81.09</td><td>3.34x</td></tr>
</table>

## Unmarshaler

<table>
<tr><td>benchmark</td><td>old ns/op</td><td>new ns/op</td><td>delta</td></tr>
<tr><td>BenchmarkNidOptNativeProtoUnmarshal</td><td>2521</td><td>1006</td><td>-60.10%</td></tr>
<tr><td>BenchmarkNinOptNativeProtoUnmarshal</td><td>2529</td><td>1750</td><td>-30.80%</td></tr>
<tr><td>BenchmarkNidRepNativeProtoUnmarshal</td><td>49067</td><td>35299</td><td>-28.06%</td></tr>
<tr><td>BenchmarkNinRepNativeProtoUnmarshal</td><td>47990</td><td>35456</td><td>-26.12%</td></tr>
<tr><td>BenchmarkNidRepPackedNativeProtoUnmarshal</td><td>26456</td><td>23950</td><td>-9.47%</td></tr>
<tr><td>BenchmarkNinRepPackedNativeProtoUnmarshal</td><td>26499</td><td>24037</td><td>-9.29%</td></tr>
<tr><td>BenchmarkNidOptStructProtoUnmarshal</td><td>6803</td><td>3873</td><td>-43.07%</td></tr>
<tr><td>BenchmarkNinOptStructProtoUnmarshal</td><td>6786</td><td>4154</td><td>-38.79%</td></tr>
<tr><td>BenchmarkNidRepStructProtoUnmarshal</td><td>56276</td><td>31970</td><td>-43.19%</td></tr>
<tr><td>BenchmarkNinRepStructProtoUnmarshal</td><td>48750</td><td>31832</td><td>-34.70%</td></tr>
<tr><td>BenchmarkNidEmbeddedStructProtoUnmarshal</td><td>4556</td><td>1973</td><td>-56.69%</td></tr>
<tr><td>BenchmarkNinEmbeddedStructProtoUnmarshal</td><td>4485</td><td>1975</td><td>-55.96%</td></tr>
<tr><td>BenchmarkNidNestedStructProtoUnmarshal</td><td>223395</td><td>135844</td><td>-39.19%</td></tr>
<tr><td>BenchmarkNinNestedStructProtoUnmarshal</td><td>226446</td><td>134022</td><td>-40.82%</td></tr>
<tr><td>BenchmarkNidOptCustomProtoUnmarshal</td><td>1859</td><td>300</td><td>-83.86%</td></tr>
<tr><td>BenchmarkNinOptCustomProtoUnmarshal</td><td>1486</td><td>402</td><td>-72.95%</td></tr>
<tr><td>BenchmarkNidRepCustomProtoUnmarshal</td><td>8229</td><td>1669</td><td>-79.72%</td></tr>
<tr><td>BenchmarkNinRepCustomProtoUnmarshal</td><td>8253</td><td>1649</td><td>-80.02%</td></tr>
<tr><td>BenchmarkNinOptNativeUnionProtoUnmarshal</td><td>840</td><td>307</td><td>-63.45%</td></tr>
<tr><td>BenchmarkNinOptStructUnionProtoUnmarshal</td><td>1395</td><td>639</td><td>-54.19%</td></tr>
<tr><td>BenchmarkNinEmbeddedStructUnionProtoUnmarshal</td><td>2297</td><td>1167</td><td>-49.19%</td></tr>
<tr><td>BenchmarkNinNestedStructUnionProtoUnmarshal</td><td>1820</td><td>889</td><td>-51.15%</td></tr>
<tr><td>BenchmarkTreeProtoUnmarshal</td><td>1521</td><td>720</td><td>-52.66%</td></tr>
<tr><td>BenchmarkOrBranchProtoUnmarshal</td><td>2669</td><td>1385</td><td>-48.11%</td></tr>
<tr><td>BenchmarkAndBranchProtoUnmarshal</td><td>2667</td><td>1420</td><td>-46.76%</td></tr>
<tr><td>BenchmarkLeafProtoUnmarshal</td><td>1171</td><td>584</td><td>-50.13%</td></tr>
<tr><td>BenchmarkDeepTreeProtoUnmarshal</td><td>2065</td><td>1081</td><td>-47.65%</td></tr>
<tr><td>BenchmarkADeepBranchProtoUnmarshal</td><td>2695</td><td>1178</td><td>-56.29%</td></tr>
<tr><td>BenchmarkAndDeepBranchProtoUnmarshal</td><td>4055</td><td>1918</td><td>-52.70%</td></tr>
<tr><td>BenchmarkDeepLeafProtoUnmarshal</td><td>1758</td><td>865</td><td>-50.80%</td></tr>
<tr><td>BenchmarkNilProtoUnmarshal</td><td>564</td><td>63</td><td>-88.79%</td></tr>
<tr><td>BenchmarkNidOptEnumProtoUnmarshal</td><td>762</td><td>73</td><td>-90.34%</td></tr>
<tr><td>BenchmarkNinOptEnumProtoUnmarshal</td><td>764</td><td>163</td><td>-78.66%</td></tr>
<tr><td>BenchmarkNidRepEnumProtoUnmarshal</td><td>1078</td><td>447</td><td>-58.53%</td></tr>
<tr><td>BenchmarkNinRepEnumProtoUnmarshal</td><td>1071</td><td>479</td><td>-55.28%</td></tr>
<tr><td>BenchmarkTimerProtoUnmarshal</td><td>1128</td><td>362</td><td>-67.91%</td></tr>
<tr><td>BenchmarkMyExtendableProtoUnmarshal</td><td>808</td><td>217</td><td>-73.14%</td></tr>
<tr><td>BenchmarkOtherExtenableProtoUnmarshal</td><td>1233</td><td>517</td><td>-58.07%</td></tr>
</table>

<table>
<tr><td>benchmark</td><td>old MB/s</td><td>new MB/s</td><td>speedup</td></tr>
<tr><td>BenchmarkNidOptNativeProtoUnmarshal</td><td>133.67</td><td>334.98</td><td>2.51x</td></tr>
<tr><td>BenchmarkNinOptNativeProtoUnmarshal</td><td>119.77</td><td>173.08</td><td>1.45x</td></tr>
<tr><td>BenchmarkNidRepNativeProtoUnmarshal</td><td>143.23</td><td>199.12</td><td>1.39x</td></tr>
<tr><td>BenchmarkNinRepNativeProtoUnmarshal</td><td>146.07</td><td>198.16</td><td>1.36x</td></tr>
<tr><td>BenchmarkNidRepPackedNativeProtoUnmarshal</td><td>127.80</td><td>141.04</td><td>1.10x</td></tr>
<tr><td>BenchmarkNinRepPackedNativeProtoUnmarshal</td><td>127.55</td><td>140.78</td><td>1.10x</td></tr>
<tr><td>BenchmarkNidOptStructProtoUnmarshal</td><td>185.79</td><td>326.31</td><td>1.76x</td></tr>
<tr><td>BenchmarkNinOptStructProtoUnmarshal</td><td>167.68</td><td>273.66</td><td>1.63x</td></tr>
<tr><td>BenchmarkNidRepStructProtoUnmarshal</td><td>147.88</td><td>260.39</td><td>1.76x</td></tr>
<tr><td>BenchmarkNinRepStructProtoUnmarshal</td><td>171.20</td><td>261.97</td><td>1.53x</td></tr>
<tr><td>BenchmarkNidEmbeddedStructProtoUnmarshal</td><td>142.86</td><td>329.42</td><td>2.31x</td></tr>
<tr><td>BenchmarkNinEmbeddedStructProtoUnmarshal</td><td>137.33</td><td>311.83</td><td>2.27x</td></tr>
<tr><td>BenchmarkNidNestedStructProtoUnmarshal</td><td>154.97</td><td>259.47</td><td>1.67x</td></tr>
<tr><td>BenchmarkNinNestedStructProtoUnmarshal</td><td>154.32</td><td>258.42</td><td>1.67x</td></tr>
<tr><td>BenchmarkNidOptCustomProtoUnmarshal</td><td>19.36</td><td>119.66</td><td>6.18x</td></tr>
<tr><td>BenchmarkNinOptCustomProtoUnmarshal</td><td>21.52</td><td>79.50</td><td>3.69x</td></tr>
<tr><td>BenchmarkNidRepCustomProtoUnmarshal</td><td>17.62</td><td>86.86</td><td>4.93x</td></tr>
<tr><td>BenchmarkNinRepCustomProtoUnmarshal</td><td>17.57</td><td>87.92</td><td>5.00x</td></tr>
<tr><td>BenchmarkNinOptNativeUnionProtoUnmarshal</td><td>38.07</td><td>104.12</td><td>2.73x</td></tr>
<tr><td>BenchmarkNinOptStructUnionProtoUnmarshal</td><td>73.08</td><td>159.54</td><td>2.18x</td></tr>
<tr><td>BenchmarkNinEmbeddedStructUnionProtoUnmarshal</td><td>94.00</td><td>185.92</td><td>1.98x</td></tr>
<tr><td>BenchmarkNinNestedStructUnionProtoUnmarshal</td><td>65.35</td><td>133.75</td><td>2.05x</td></tr>
<tr><td>BenchmarkTreeProtoUnmarshal</td><td>141.28</td><td>297.13</td><td>2.10x</td></tr>
<tr><td>BenchmarkOrBranchProtoUnmarshal</td><td>162.56</td><td>313.96</td><td>1.93x</td></tr>
<tr><td>BenchmarkAndBranchProtoUnmarshal</td><td>163.06</td><td>306.15</td><td>1.88x</td></tr>
<tr><td>BenchmarkLeafProtoUnmarshal</td><td>176.72</td><td>354.19</td><td>2.00x</td></tr>
<tr><td>BenchmarkDeepTreeProtoUnmarshal</td><td>107.50</td><td>205.30</td><td>1.91x</td></tr>
<tr><td>BenchmarkADeepBranchProtoUnmarshal</td><td>83.48</td><td>190.88</td><td>2.29x</td></tr>
<tr><td>BenchmarkAndDeepBranchProtoUnmarshal</td><td>110.97</td><td>234.60</td><td>2.11x</td></tr>
<tr><td>BenchmarkDeepLeafProtoUnmarshal</td><td>123.40</td><td>250.73</td><td>2.03x</td></tr>
<tr><td>BenchmarkNidOptEnumProtoUnmarshal</td><td>2.62</td><td>27.16</td><td>10.37x</td></tr>
<tr><td>BenchmarkNinOptEnumProtoUnmarshal</td><td>1.31</td><td>6.11</td><td>4.66x</td></tr>
<tr><td>BenchmarkNidRepEnumProtoUnmarshal</td><td>7.42</td><td>17.88</td><td>2.41x</td></tr>
<tr><td>BenchmarkNinRepEnumProtoUnmarshal</td><td>7.47</td><td>16.69</td><td>2.23x</td></tr>
<tr><td>BenchmarkTimerProtoUnmarshal</td><td>61.12</td><td>190.34</td><td>3.11x</td></tr>
<tr><td>BenchmarkMyExtendableProtoUnmarshal</td><td>9.90</td><td>36.71</td><td>3.71x</td></tr>
<tr><td>BenchmarkOtherExtenableProtoUnmarshal</td><td>21.90</td><td>52.13</td><td>2.38x</td></tr>
</table>