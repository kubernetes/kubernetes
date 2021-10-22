package dynamodbattribute_test

import (
	"fmt"
	"reflect"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
)

func ExampleMarshal() {
	type Record struct {
		Bytes   []byte
		MyField string
		Letters []string
		Numbers []int
	}

	r := Record{
		Bytes:   []byte{48, 49},
		MyField: "MyFieldValue",
		Letters: []string{"a", "b", "c", "d"},
		Numbers: []int{1, 2, 3},
	}
	av, err := dynamodbattribute.Marshal(r)
	fmt.Println("err", err)
	fmt.Println("Bytes", av.M["Bytes"])
	fmt.Println("MyField", av.M["MyField"])
	fmt.Println("Letters", av.M["Letters"])
	fmt.Println("Numbers", av.M["Numbers"])

	// Output:
	// err <nil>
	// Bytes {
	//   B: <binary> len 2
	// }
	// MyField {
	//   S: "MyFieldValue"
	// }
	// Letters {
	//   L: [
	//     {
	//       S: "a"
	//     },
	//     {
	//       S: "b"
	//     },
	//     {
	//       S: "c"
	//     },
	//     {
	//       S: "d"
	//     }
	//   ]
	// }
	// Numbers {
	//   L: [{
	//       N: "1"
	//     },{
	//       N: "2"
	//     },{
	//       N: "3"
	//     }]
	// }
}

func ExampleUnmarshal() {
	type Record struct {
		Bytes   []byte
		MyField string
		Letters []string
		A2Num   map[string]int
	}

	expect := Record{
		Bytes:   []byte{48, 49},
		MyField: "MyFieldValue",
		Letters: []string{"a", "b", "c", "d"},
		A2Num:   map[string]int{"a": 1, "b": 2, "c": 3},
	}

	av := &dynamodb.AttributeValue{
		M: map[string]*dynamodb.AttributeValue{
			"Bytes":   {B: []byte{48, 49}},
			"MyField": {S: aws.String("MyFieldValue")},
			"Letters": {L: []*dynamodb.AttributeValue{
				{S: aws.String("a")}, {S: aws.String("b")}, {S: aws.String("c")}, {S: aws.String("d")},
			}},
			"A2Num": {M: map[string]*dynamodb.AttributeValue{
				"a": {N: aws.String("1")},
				"b": {N: aws.String("2")},
				"c": {N: aws.String("3")},
			}},
		},
	}

	actual := Record{}
	err := dynamodbattribute.Unmarshal(av, &actual)
	fmt.Println(err, reflect.DeepEqual(expect, actual))

	// Output:
	// <nil> true
}
