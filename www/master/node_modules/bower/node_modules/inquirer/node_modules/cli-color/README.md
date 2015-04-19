# cli-color
## Yet another colors and formatting for the console solution

Colors, formatting and other goodies for the console. This package won't mess with built-ins and provides neat way to predefine formatting patterns, see below.

## Installation

	$ npm install cli-color

## Usage

Usage:

```javascript
var clc = require('cli-color');
```

Output colored text:

```javascript
console.log(clc.red('Text in red'));
```

Styles can be mixed:

```javascript
console.log(clc.red.bgWhite.underline('Underlined red text on white background.'));
```

Styled text can be mixed with unstyled:

```javascript
console.log(clc.red('red') + ' plain ' + clc.blue('blue'));
```

__Best way is to predefine needed stylings and then use it__:

```javascript
var error = clc.red.bold;
var warn = clc.yellow;
var notice = clc.blue;

console.log(error('Error!'));
console.log(warn('Warning'));
console.log(notice('Notice'));
```

Supported are all ANSI colors and styles:

#### Styles

Styles will display correctly if font used in your console supports them.

* bold
* italic
* underline
* blink
* inverse
* strike

#### Colors

<table>
  <thead><th>Foreground</th><th>Background</th><th></th></thead>
  <tbody>
    <tr><td>black</td><td>bgBlack</td><td><img src="http://medyk.org/colors/000000.png" width="30" height="30" /></td></tr>
    <tr><td>red</td><td>bgRed</td><td><img src="http://medyk.org/colors/800000.png" width="30" height="30" /></td></tr>
    <tr><td>green</td><td>bgGreen</td><td><img src="http://medyk.org/colors/008000.png" width="30" height="30" /></td></tr>
    <tr><td>yellow</td><td>bgYellow</td><td><img src="http://medyk.org/colors/808000.png" width="30" height="30" /></td></tr>
    <tr><td>blue</td><td>bgBlue</td><td><img src="http://medyk.org/colors/000080.png" width="30" height="30" /></td></tr>
    <tr><td>magenta</td><td>bgMagenta</td><td><img src="http://medyk.org/colors/800080.png" width="30" height="30" /></td></tr>
    <tr><td>cyan</td><td>bgCyan</td><td><img src="http://medyk.org/colors/008080.png" width="30" height="30" /></td></tr>
    <tr><td>white</td><td>bgWhite</td><td><img src="http://medyk.org/colors/c0c0c0.png" width="30" height="30" /></td></tr>
  </tbody>
</table>

##### Bright variants

<table>
  <thead><th>Foreground</th><th>Background</th><th></th></thead>
  <tbody>
    <tr><td>blackBright</td><td>bgBlackBright</td><td><img src="http://medyk.org/colors/808080.png" width="30" height="30" /></td></tr>
    <tr><td>redBright</td><td>bgRedBright</td><td><img src="http://medyk.org/colors/ff0000.png" width="30" height="30" /></td></tr>
    <tr><td>greenBright</td><td>bgGreenBright</td><td><img src="http://medyk.org/colors/00ff00.png" width="30" height="30" /></td></tr>
    <tr><td>yellowBright</td><td>bgYellowBright</td><td><img src="http://medyk.org/colors/ffff00.png" width="30" height="30" /></td></tr>
    <tr><td>blueBright</td><td>bgBlueBright</td><td><img src="http://medyk.org/colors/0000ff.png" width="30" height="30" /></td></tr>
    <tr><td>magentaBright</td><td>bgMagentaBright</td><td><img src="http://medyk.org/colors/ff00ff.png" width="30" height="30" /></td></tr>
    <tr><td>cyanBright</td><td>bgCyanBright</td><td><img src="http://medyk.org/colors/00ffff.png" width="30" height="30" /></td></tr>
    <tr><td>whiteBright</td><td>bgWhiteBright</td><td><img src="http://medyk.org/colors/ffffff.png" width="30" height="30" /></td></tr>
  </tbody>
</table>

##### xTerm colors (256 colors table)

__Not supported on Windows and some terminals__. However if used in not supported environment, the closest color from basic (16 colors) palette is chosen.

Usage:

```javascript
var msg = clc.xterm(202).bgXterm(236);
console.log(msg('Orange text on dark gray background'));
```

Color table:

<table>
  <tr>
    <td>0</td><td><img src="http://medyk.org/colors/000000.png" width="20" height="20" /></td>
    <td>1</td><td><img src="http://medyk.org/colors/800000.png" width="20" height="20" /></td>
    <td>2</td><td><img src="http://medyk.org/colors/008000.png" width="20" height="20" /></td>
    <td>3</td><td><img src="http://medyk.org/colors/808000.png" width="20" height="20" /></td>
    <td>4</td><td><img src="http://medyk.org/colors/000080.png" width="20" height="20" /></td>
    <td>5</td><td><img src="http://medyk.org/colors/800080.png" width="20" height="20" /></td>
    <td>6</td><td><img src="http://medyk.org/colors/008080.png" width="20" height="20" /></td>
    <td>7</td><td><img src="http://medyk.org/colors/c0c0c0.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>8</td><td><img src="http://medyk.org/colors/808080.png" width="20" height="20" /></td>
    <td>9</td><td><img src="http://medyk.org/colors/ff0000.png" width="20" height="20" /></td>
    <td>10</td><td><img src="http://medyk.org/colors/00ff00.png" width="20" height="20" /></td>
    <td>11</td><td><img src="http://medyk.org/colors/ffff00.png" width="20" height="20" /></td>
    <td>12</td><td><img src="http://medyk.org/colors/0000ff.png" width="20" height="20" /></td>
    <td>13</td><td><img src="http://medyk.org/colors/ff00ff.png" width="20" height="20" /></td>
    <td>14</td><td><img src="http://medyk.org/colors/00ffff.png" width="20" height="20" /></td>
    <td>15</td><td><img src="http://medyk.org/colors/ffffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>16</td><td><img src="http://medyk.org/colors/000000.png" width="20" height="20" /></td>
    <td>17</td><td><img src="http://medyk.org/colors/00005f.png" width="20" height="20" /></td>
    <td>18</td><td><img src="http://medyk.org/colors/000087.png" width="20" height="20" /></td>
    <td>19</td><td><img src="http://medyk.org/colors/0000af.png" width="20" height="20" /></td>
    <td>20</td><td><img src="http://medyk.org/colors/0000d7.png" width="20" height="20" /></td>
    <td>21</td><td><img src="http://medyk.org/colors/0000ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>22</td><td><img src="http://medyk.org/colors/005f00.png" width="20" height="20" /></td>
    <td>23</td><td><img src="http://medyk.org/colors/005f5f.png" width="20" height="20" /></td>
    <td>24</td><td><img src="http://medyk.org/colors/005f87.png" width="20" height="20" /></td>
    <td>25</td><td><img src="http://medyk.org/colors/005faf.png" width="20" height="20" /></td>
    <td>26</td><td><img src="http://medyk.org/colors/005fd7.png" width="20" height="20" /></td>
    <td>27</td><td><img src="http://medyk.org/colors/005fff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>28</td><td><img src="http://medyk.org/colors/008700.png" width="20" height="20" /></td>
    <td>29</td><td><img src="http://medyk.org/colors/00875f.png" width="20" height="20" /></td>
    <td>30</td><td><img src="http://medyk.org/colors/008787.png" width="20" height="20" /></td>
    <td>31</td><td><img src="http://medyk.org/colors/0087af.png" width="20" height="20" /></td>
    <td>32</td><td><img src="http://medyk.org/colors/0087d7.png" width="20" height="20" /></td>
    <td>33</td><td><img src="http://medyk.org/colors/0087ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>34</td><td><img src="http://medyk.org/colors/00af00.png" width="20" height="20" /></td>
    <td>35</td><td><img src="http://medyk.org/colors/00af5f.png" width="20" height="20" /></td>
    <td>36</td><td><img src="http://medyk.org/colors/00af87.png" width="20" height="20" /></td>
    <td>37</td><td><img src="http://medyk.org/colors/00afaf.png" width="20" height="20" /></td>
    <td>38</td><td><img src="http://medyk.org/colors/00afd7.png" width="20" height="20" /></td>
    <td>39</td><td><img src="http://medyk.org/colors/00afff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>40</td><td><img src="http://medyk.org/colors/00d700.png" width="20" height="20" /></td>
    <td>41</td><td><img src="http://medyk.org/colors/00d75f.png" width="20" height="20" /></td>
    <td>42</td><td><img src="http://medyk.org/colors/00d787.png" width="20" height="20" /></td>
    <td>43</td><td><img src="http://medyk.org/colors/00d7af.png" width="20" height="20" /></td>
    <td>44</td><td><img src="http://medyk.org/colors/00d7d7.png" width="20" height="20" /></td>
    <td>45</td><td><img src="http://medyk.org/colors/00d7ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>46</td><td><img src="http://medyk.org/colors/00ff00.png" width="20" height="20" /></td>
    <td>47</td><td><img src="http://medyk.org/colors/00ff5f.png" width="20" height="20" /></td>
    <td>48</td><td><img src="http://medyk.org/colors/00ff87.png" width="20" height="20" /></td>
    <td>49</td><td><img src="http://medyk.org/colors/00ffaf.png" width="20" height="20" /></td>
    <td>50</td><td><img src="http://medyk.org/colors/00ffd7.png" width="20" height="20" /></td>
    <td>51</td><td><img src="http://medyk.org/colors/00ffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>52</td><td><img src="http://medyk.org/colors/5f0000.png" width="20" height="20" /></td>
    <td>53</td><td><img src="http://medyk.org/colors/5f005f.png" width="20" height="20" /></td>
    <td>54</td><td><img src="http://medyk.org/colors/5f0087.png" width="20" height="20" /></td>
    <td>55</td><td><img src="http://medyk.org/colors/5f00af.png" width="20" height="20" /></td>
    <td>56</td><td><img src="http://medyk.org/colors/5f00d7.png" width="20" height="20" /></td>
    <td>57</td><td><img src="http://medyk.org/colors/5f00ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>58</td><td><img src="http://medyk.org/colors/5f5f00.png" width="20" height="20" /></td>
    <td>59</td><td><img src="http://medyk.org/colors/5f5f5f.png" width="20" height="20" /></td>
    <td>60</td><td><img src="http://medyk.org/colors/5f5f87.png" width="20" height="20" /></td>
    <td>61</td><td><img src="http://medyk.org/colors/5f5faf.png" width="20" height="20" /></td>
    <td>62</td><td><img src="http://medyk.org/colors/5f5fd7.png" width="20" height="20" /></td>
    <td>63</td><td><img src="http://medyk.org/colors/5f5fff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>64</td><td><img src="http://medyk.org/colors/5f8700.png" width="20" height="20" /></td>
    <td>65</td><td><img src="http://medyk.org/colors/5f875f.png" width="20" height="20" /></td>
    <td>66</td><td><img src="http://medyk.org/colors/5f8787.png" width="20" height="20" /></td>
    <td>67</td><td><img src="http://medyk.org/colors/5f87af.png" width="20" height="20" /></td>
    <td>68</td><td><img src="http://medyk.org/colors/5f87d7.png" width="20" height="20" /></td>
    <td>69</td><td><img src="http://medyk.org/colors/5f87ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>70</td><td><img src="http://medyk.org/colors/5faf00.png" width="20" height="20" /></td>
    <td>71</td><td><img src="http://medyk.org/colors/5faf5f.png" width="20" height="20" /></td>
    <td>72</td><td><img src="http://medyk.org/colors/5faf87.png" width="20" height="20" /></td>
    <td>73</td><td><img src="http://medyk.org/colors/5fafaf.png" width="20" height="20" /></td>
    <td>74</td><td><img src="http://medyk.org/colors/5fafd7.png" width="20" height="20" /></td>
    <td>75</td><td><img src="http://medyk.org/colors/5fafff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>76</td><td><img src="http://medyk.org/colors/5fd700.png" width="20" height="20" /></td>
    <td>77</td><td><img src="http://medyk.org/colors/5fd75f.png" width="20" height="20" /></td>
    <td>78</td><td><img src="http://medyk.org/colors/5fd787.png" width="20" height="20" /></td>
    <td>79</td><td><img src="http://medyk.org/colors/5fd7af.png" width="20" height="20" /></td>
    <td>80</td><td><img src="http://medyk.org/colors/5fd7d7.png" width="20" height="20" /></td>
    <td>81</td><td><img src="http://medyk.org/colors/5fd7ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>82</td><td><img src="http://medyk.org/colors/5fff00.png" width="20" height="20" /></td>
    <td>83</td><td><img src="http://medyk.org/colors/5fff5f.png" width="20" height="20" /></td>
    <td>84</td><td><img src="http://medyk.org/colors/5fff87.png" width="20" height="20" /></td>
    <td>85</td><td><img src="http://medyk.org/colors/5fffaf.png" width="20" height="20" /></td>
    <td>86</td><td><img src="http://medyk.org/colors/5fffd7.png" width="20" height="20" /></td>
    <td>87</td><td><img src="http://medyk.org/colors/5fffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>88</td><td><img src="http://medyk.org/colors/870000.png" width="20" height="20" /></td>
    <td>89</td><td><img src="http://medyk.org/colors/87005f.png" width="20" height="20" /></td>
    <td>90</td><td><img src="http://medyk.org/colors/870087.png" width="20" height="20" /></td>
    <td>91</td><td><img src="http://medyk.org/colors/8700af.png" width="20" height="20" /></td>
    <td>92</td><td><img src="http://medyk.org/colors/8700d7.png" width="20" height="20" /></td>
    <td>93</td><td><img src="http://medyk.org/colors/8700ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>94</td><td><img src="http://medyk.org/colors/875f00.png" width="20" height="20" /></td>
    <td>95</td><td><img src="http://medyk.org/colors/875f5f.png" width="20" height="20" /></td>
    <td>96</td><td><img src="http://medyk.org/colors/875f87.png" width="20" height="20" /></td>
    <td>97</td><td><img src="http://medyk.org/colors/875faf.png" width="20" height="20" /></td>
    <td>98</td><td><img src="http://medyk.org/colors/875fd7.png" width="20" height="20" /></td>
    <td>99</td><td><img src="http://medyk.org/colors/875fff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>100</td><td><img src="http://medyk.org/colors/878700.png" width="20" height="20" /></td>
    <td>101</td><td><img src="http://medyk.org/colors/87875f.png" width="20" height="20" /></td>
    <td>102</td><td><img src="http://medyk.org/colors/878787.png" width="20" height="20" /></td>
    <td>103</td><td><img src="http://medyk.org/colors/8787af.png" width="20" height="20" /></td>
    <td>104</td><td><img src="http://medyk.org/colors/8787d7.png" width="20" height="20" /></td>
    <td>105</td><td><img src="http://medyk.org/colors/8787ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>106</td><td><img src="http://medyk.org/colors/87af00.png" width="20" height="20" /></td>
    <td>107</td><td><img src="http://medyk.org/colors/87af5f.png" width="20" height="20" /></td>
    <td>108</td><td><img src="http://medyk.org/colors/87af87.png" width="20" height="20" /></td>
    <td>109</td><td><img src="http://medyk.org/colors/87afaf.png" width="20" height="20" /></td>
    <td>110</td><td><img src="http://medyk.org/colors/87afd7.png" width="20" height="20" /></td>
    <td>111</td><td><img src="http://medyk.org/colors/87afff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>112</td><td><img src="http://medyk.org/colors/87d700.png" width="20" height="20" /></td>
    <td>113</td><td><img src="http://medyk.org/colors/87d75f.png" width="20" height="20" /></td>
    <td>114</td><td><img src="http://medyk.org/colors/87d787.png" width="20" height="20" /></td>
    <td>115</td><td><img src="http://medyk.org/colors/87d7af.png" width="20" height="20" /></td>
    <td>116</td><td><img src="http://medyk.org/colors/87d7d7.png" width="20" height="20" /></td>
    <td>117</td><td><img src="http://medyk.org/colors/87d7ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>118</td><td><img src="http://medyk.org/colors/87ff00.png" width="20" height="20" /></td>
    <td>119</td><td><img src="http://medyk.org/colors/87ff5f.png" width="20" height="20" /></td>
    <td>120</td><td><img src="http://medyk.org/colors/87ff87.png" width="20" height="20" /></td>
    <td>121</td><td><img src="http://medyk.org/colors/87ffaf.png" width="20" height="20" /></td>
    <td>122</td><td><img src="http://medyk.org/colors/87ffd7.png" width="20" height="20" /></td>
    <td>123</td><td><img src="http://medyk.org/colors/87ffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>124</td><td><img src="http://medyk.org/colors/af0000.png" width="20" height="20" /></td>
    <td>125</td><td><img src="http://medyk.org/colors/af005f.png" width="20" height="20" /></td>
    <td>126</td><td><img src="http://medyk.org/colors/af0087.png" width="20" height="20" /></td>
    <td>127</td><td><img src="http://medyk.org/colors/af00af.png" width="20" height="20" /></td>
    <td>128</td><td><img src="http://medyk.org/colors/af00d7.png" width="20" height="20" /></td>
    <td>129</td><td><img src="http://medyk.org/colors/af00ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>130</td><td><img src="http://medyk.org/colors/af5f00.png" width="20" height="20" /></td>
    <td>131</td><td><img src="http://medyk.org/colors/af5f5f.png" width="20" height="20" /></td>
    <td>132</td><td><img src="http://medyk.org/colors/af5f87.png" width="20" height="20" /></td>
    <td>133</td><td><img src="http://medyk.org/colors/af5faf.png" width="20" height="20" /></td>
    <td>134</td><td><img src="http://medyk.org/colors/af5fd7.png" width="20" height="20" /></td>
    <td>135</td><td><img src="http://medyk.org/colors/af5fff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>136</td><td><img src="http://medyk.org/colors/af8700.png" width="20" height="20" /></td>
    <td>137</td><td><img src="http://medyk.org/colors/af875f.png" width="20" height="20" /></td>
    <td>138</td><td><img src="http://medyk.org/colors/af8787.png" width="20" height="20" /></td>
    <td>139</td><td><img src="http://medyk.org/colors/af87af.png" width="20" height="20" /></td>
    <td>140</td><td><img src="http://medyk.org/colors/af87d7.png" width="20" height="20" /></td>
    <td>141</td><td><img src="http://medyk.org/colors/af87ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>142</td><td><img src="http://medyk.org/colors/afaf00.png" width="20" height="20" /></td>
    <td>143</td><td><img src="http://medyk.org/colors/afaf5f.png" width="20" height="20" /></td>
    <td>144</td><td><img src="http://medyk.org/colors/afaf87.png" width="20" height="20" /></td>
    <td>145</td><td><img src="http://medyk.org/colors/afafaf.png" width="20" height="20" /></td>
    <td>146</td><td><img src="http://medyk.org/colors/afafd7.png" width="20" height="20" /></td>
    <td>147</td><td><img src="http://medyk.org/colors/afafff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>148</td><td><img src="http://medyk.org/colors/afd700.png" width="20" height="20" /></td>
    <td>149</td><td><img src="http://medyk.org/colors/afd75f.png" width="20" height="20" /></td>
    <td>150</td><td><img src="http://medyk.org/colors/afd787.png" width="20" height="20" /></td>
    <td>151</td><td><img src="http://medyk.org/colors/afd7af.png" width="20" height="20" /></td>
    <td>152</td><td><img src="http://medyk.org/colors/afd7d7.png" width="20" height="20" /></td>
    <td>153</td><td><img src="http://medyk.org/colors/afd7ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>154</td><td><img src="http://medyk.org/colors/afff00.png" width="20" height="20" /></td>
    <td>155</td><td><img src="http://medyk.org/colors/afff5f.png" width="20" height="20" /></td>
    <td>156</td><td><img src="http://medyk.org/colors/afff87.png" width="20" height="20" /></td>
    <td>157</td><td><img src="http://medyk.org/colors/afffaf.png" width="20" height="20" /></td>
    <td>158</td><td><img src="http://medyk.org/colors/afffd7.png" width="20" height="20" /></td>
    <td>159</td><td><img src="http://medyk.org/colors/afffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>160</td><td><img src="http://medyk.org/colors/d70000.png" width="20" height="20" /></td>
    <td>161</td><td><img src="http://medyk.org/colors/d7005f.png" width="20" height="20" /></td>
    <td>162</td><td><img src="http://medyk.org/colors/d70087.png" width="20" height="20" /></td>
    <td>163</td><td><img src="http://medyk.org/colors/d700af.png" width="20" height="20" /></td>
    <td>164</td><td><img src="http://medyk.org/colors/d700d7.png" width="20" height="20" /></td>
    <td>165</td><td><img src="http://medyk.org/colors/d700ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>166</td><td><img src="http://medyk.org/colors/d75f00.png" width="20" height="20" /></td>
    <td>167</td><td><img src="http://medyk.org/colors/d75f5f.png" width="20" height="20" /></td>
    <td>168</td><td><img src="http://medyk.org/colors/d75f87.png" width="20" height="20" /></td>
    <td>169</td><td><img src="http://medyk.org/colors/d75faf.png" width="20" height="20" /></td>
    <td>170</td><td><img src="http://medyk.org/colors/d75fd7.png" width="20" height="20" /></td>
    <td>171</td><td><img src="http://medyk.org/colors/d75fff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>172</td><td><img src="http://medyk.org/colors/d78700.png" width="20" height="20" /></td>
    <td>173</td><td><img src="http://medyk.org/colors/d7875f.png" width="20" height="20" /></td>
    <td>174</td><td><img src="http://medyk.org/colors/d78787.png" width="20" height="20" /></td>
    <td>175</td><td><img src="http://medyk.org/colors/d787af.png" width="20" height="20" /></td>
    <td>176</td><td><img src="http://medyk.org/colors/d787d7.png" width="20" height="20" /></td>
    <td>177</td><td><img src="http://medyk.org/colors/d787ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>178</td><td><img src="http://medyk.org/colors/d7af00.png" width="20" height="20" /></td>
    <td>179</td><td><img src="http://medyk.org/colors/d7af5f.png" width="20" height="20" /></td>
    <td>180</td><td><img src="http://medyk.org/colors/d7af87.png" width="20" height="20" /></td>
    <td>181</td><td><img src="http://medyk.org/colors/d7afaf.png" width="20" height="20" /></td>
    <td>182</td><td><img src="http://medyk.org/colors/d7afd7.png" width="20" height="20" /></td>
    <td>183</td><td><img src="http://medyk.org/colors/d7afff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>184</td><td><img src="http://medyk.org/colors/d7d700.png" width="20" height="20" /></td>
    <td>185</td><td><img src="http://medyk.org/colors/d7d75f.png" width="20" height="20" /></td>
    <td>186</td><td><img src="http://medyk.org/colors/d7d787.png" width="20" height="20" /></td>
    <td>187</td><td><img src="http://medyk.org/colors/d7d7af.png" width="20" height="20" /></td>
    <td>188</td><td><img src="http://medyk.org/colors/d7d7d7.png" width="20" height="20" /></td>
    <td>189</td><td><img src="http://medyk.org/colors/d7d7ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>190</td><td><img src="http://medyk.org/colors/d7ff00.png" width="20" height="20" /></td>
    <td>191</td><td><img src="http://medyk.org/colors/d7ff5f.png" width="20" height="20" /></td>
    <td>192</td><td><img src="http://medyk.org/colors/d7ff87.png" width="20" height="20" /></td>
    <td>193</td><td><img src="http://medyk.org/colors/d7ffaf.png" width="20" height="20" /></td>
    <td>194</td><td><img src="http://medyk.org/colors/d7ffd7.png" width="20" height="20" /></td>
    <td>195</td><td><img src="http://medyk.org/colors/d7ffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>196</td><td><img src="http://medyk.org/colors/ff0000.png" width="20" height="20" /></td>
    <td>197</td><td><img src="http://medyk.org/colors/ff005f.png" width="20" height="20" /></td>
    <td>198</td><td><img src="http://medyk.org/colors/ff0087.png" width="20" height="20" /></td>
    <td>199</td><td><img src="http://medyk.org/colors/ff00af.png" width="20" height="20" /></td>
    <td>200</td><td><img src="http://medyk.org/colors/ff00d7.png" width="20" height="20" /></td>
    <td>201</td><td><img src="http://medyk.org/colors/ff00ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>202</td><td><img src="http://medyk.org/colors/ff5f00.png" width="20" height="20" /></td>
    <td>203</td><td><img src="http://medyk.org/colors/ff5f5f.png" width="20" height="20" /></td>
    <td>204</td><td><img src="http://medyk.org/colors/ff5f87.png" width="20" height="20" /></td>
    <td>205</td><td><img src="http://medyk.org/colors/ff5faf.png" width="20" height="20" /></td>
    <td>206</td><td><img src="http://medyk.org/colors/ff5fd7.png" width="20" height="20" /></td>
    <td>207</td><td><img src="http://medyk.org/colors/ff5fff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>208</td><td><img src="http://medyk.org/colors/ff8700.png" width="20" height="20" /></td>
    <td>209</td><td><img src="http://medyk.org/colors/ff875f.png" width="20" height="20" /></td>
    <td>210</td><td><img src="http://medyk.org/colors/ff8787.png" width="20" height="20" /></td>
    <td>211</td><td><img src="http://medyk.org/colors/ff87af.png" width="20" height="20" /></td>
    <td>212</td><td><img src="http://medyk.org/colors/ff87d7.png" width="20" height="20" /></td>
    <td>213</td><td><img src="http://medyk.org/colors/ff87ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>214</td><td><img src="http://medyk.org/colors/ffaf00.png" width="20" height="20" /></td>
    <td>215</td><td><img src="http://medyk.org/colors/ffaf5f.png" width="20" height="20" /></td>
    <td>216</td><td><img src="http://medyk.org/colors/ffaf87.png" width="20" height="20" /></td>
    <td>217</td><td><img src="http://medyk.org/colors/ffafaf.png" width="20" height="20" /></td>
    <td>218</td><td><img src="http://medyk.org/colors/ffafd7.png" width="20" height="20" /></td>
    <td>219</td><td><img src="http://medyk.org/colors/ffafff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>220</td><td><img src="http://medyk.org/colors/ffd700.png" width="20" height="20" /></td>
    <td>221</td><td><img src="http://medyk.org/colors/ffd75f.png" width="20" height="20" /></td>
    <td>222</td><td><img src="http://medyk.org/colors/ffd787.png" width="20" height="20" /></td>
    <td>223</td><td><img src="http://medyk.org/colors/ffd7af.png" width="20" height="20" /></td>
    <td>224</td><td><img src="http://medyk.org/colors/ffd7d7.png" width="20" height="20" /></td>
    <td>225</td><td><img src="http://medyk.org/colors/ffd7ff.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>226</td><td><img src="http://medyk.org/colors/ffff00.png" width="20" height="20" /></td>
    <td>227</td><td><img src="http://medyk.org/colors/ffff5f.png" width="20" height="20" /></td>
    <td>228</td><td><img src="http://medyk.org/colors/ffff87.png" width="20" height="20" /></td>
    <td>229</td><td><img src="http://medyk.org/colors/ffffaf.png" width="20" height="20" /></td>
    <td>230</td><td><img src="http://medyk.org/colors/ffffd7.png" width="20" height="20" /></td>
    <td>231</td><td><img src="http://medyk.org/colors/ffffff.png" width="20" height="20" /></td>
  </tr>

  <tr>
    <td>232</td><td><img src="http://medyk.org/colors/080808.png" width="20" height="20" /></td>
    <td>233</td><td><img src="http://medyk.org/colors/121212.png" width="20" height="20" /></td>
    <td>234</td><td><img src="http://medyk.org/colors/1c1c1c.png" width="20" height="20" /></td>
    <td>235</td><td><img src="http://medyk.org/colors/262626.png" width="20" height="20" /></td>
    <td>236</td><td><img src="http://medyk.org/colors/303030.png" width="20" height="20" /></td>
    <td>237</td><td><img src="http://medyk.org/colors/3a3a3a.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>238</td><td><img src="http://medyk.org/colors/444444.png" width="20" height="20" /></td>
    <td>239</td><td><img src="http://medyk.org/colors/4e4e4e.png" width="20" height="20" /></td>
    <td>240</td><td><img src="http://medyk.org/colors/585858.png" width="20" height="20" /></td>
    <td>241</td><td><img src="http://medyk.org/colors/626262.png" width="20" height="20" /></td>
    <td>242</td><td><img src="http://medyk.org/colors/6c6c6c.png" width="20" height="20" /></td>
    <td>243</td><td><img src="http://medyk.org/colors/767676.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>244</td><td><img src="http://medyk.org/colors/808080.png" width="20" height="20" /></td>
    <td>245</td><td><img src="http://medyk.org/colors/8a8a8a.png" width="20" height="20" /></td>
    <td>246</td><td><img src="http://medyk.org/colors/949494.png" width="20" height="20" /></td>
    <td>247</td><td><img src="http://medyk.org/colors/9e9e9e.png" width="20" height="20" /></td>
    <td>248</td><td><img src="http://medyk.org/colors/a8a8a8.png" width="20" height="20" /></td>
    <td>249</td><td><img src="http://medyk.org/colors/b2b2b2.png" width="20" height="20" /></td>
  </tr>
  <tr>
    <td>250</td><td><img src="http://medyk.org/colors/bcbcbc.png" width="20" height="20" /></td>
    <td>251</td><td><img src="http://medyk.org/colors/c6c6c6.png" width="20" height="20" /></td>
    <td>252</td><td><img src="http://medyk.org/colors/d0d0d0.png" width="20" height="20" /></td>
    <td>253</td><td><img src="http://medyk.org/colors/dadada.png" width="20" height="20" /></td>
    <td>254</td><td><img src="http://medyk.org/colors/e4e4e4.png" width="20" height="20" /></td>
    <td>255</td><td><img src="http://medyk.org/colors/eeeeee.png" width="20" height="20" /></td>
  </tr>
</table>

#### Terminal reset

Terminal can be cleared with `clc.reset`

```javascript
console.log(clc.reset);
```

#### Move around functions

##### clc.move(x, y)

Move cursor _x_ columns and _y_ rows away. Values can be positive or negative, e.g.:

```javascript
process.stdout.write(clc.move(-2, -2)); // Move cursors two columns and two rows back
```

##### clc.moveTo(x, y)

Absolute move. Sets cursor position at _x_ column and _y_ row

```javascript
process.stdout.write(clc.moveTo(0, 0)); // Move cursor to first row and first column in terminal window
```

##### clc.bol([n[, erase]])

Move cursor to the begining of the line, with _n_ we may specify how many lines away we want to move, value can be positive or negative. Additionally we may decide to clear lines content with _erase_

```javascript
process.stdout.write(clc.bol(-2)); // Move cursor two lines back and place it at begin of the line
```

##### clc.up(n)

Move cursor up _n_ rows

##### clc.down(n)

Move cursor down _n_ rows

##### clc.right(n)

Move cursor right _n_ columns

##### clc.left(n)

Move cursor left _n_ columns

#### Terminal characteristics

##### clc.width

Returns terminal width

##### clc.height

Returns terminal height

### Additional functionalities (provided as separate modules)

#### trim(formatedText) _(cli-color/trim)_

Trims ANSI formatted string to plain text

```javascript
var ansiTrim = require('cli-color/trim');

var plain = ansiTrim(formatted);
```

#### throbber(write, interval[, format]) _(cli-color/throbber)_

Writes throbber string to _write_ function at given _interval_. Optionally throbber output can be formatted with given _format_ function

```javascript
var setupThrobber = require('cli-color/throbber');

var throbber = setupThrobber(function (str) {
  process.stdout.write(str);
}, 200);

throbber.start();

// at any time you can stop/start throbber
throbber.stop();
```

## Tests [![Build Status](https://travis-ci.org/medikoo/cli-color.png)](https://travis-ci.org/medikoo/cli-color)

	$ npm test
