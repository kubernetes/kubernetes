| json type \ dest type | bool | int | uint | float |string|
| --- | --- | --- | --- |--|--|
| number | positive => true <br/> negative => true <br/> zero => false| 23.2 => 23 <br/> -32.1 => -32| 12.1 => 12 <br/> -12.1 => 0|as normal|same as origin|
| string | empty string => false <br/> string "0" => false <br/> other strings => true | "123.32" => 123 <br/> "-123.4" => -123 <br/> "123.23xxxw" => 123 <br/>  "abcde12" => 0 <br/> "-32.1" => -32| 13.2 => 13 <br/> -1.1 => 0 |12.1 => 12.1 <br/> -12.3 => -12.3<br/> 12.4xxa => 12.4 <br/> +1.1e2 =>110 |same as origin|
| bool | true => true <br/> false => false| true => 1 <br/> false => 0 | true => 1 <br/> false => 0 |true => 1 <br/>false => 0|true => "true" <br/> false => "false"|
| object | true | 0 | 0 |0|originnal json|
| array | empty array => false <br/> nonempty array => true| [] => 0 <br/> [1,2] => 1 | [] => 0 <br/> [1,2] => 1 |[] => 0<br/>[1,2] => 1|original json|