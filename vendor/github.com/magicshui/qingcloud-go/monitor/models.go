package monitor

import (
	// "log"
	"reflect"
	"strconv"
	"strings"
)

type Meter struct {
	Data     []interface{} `json:"data"`
	VxnetID  string        `json:"vxnet_id"`
	MeterID  string        `json:"meter_id"`
	Sequence int           `json:"sequence"`
}

type Data struct {
	Role     string        `json:"role"`
	DataSet  []interface{} `json:"data_set"`
	VxnetID  string        `json:"vxnet_id"`
	MeterID  string        `json:"meter_id"`
	Sequence int           `json:"sequence"`
}

func (c *Data) UnpackData() [][]int64 {
	return unpackData(c.DataSet)
}

// TODO: 需要简化代码
func (c *Meter) UnpackData() [][]int64 {
	return unpackData(c.Data)
}

func unpackData(cData []interface{}) [][]int64 {
	var finalData = make([][]int64, 0)
	firstElemt := cData[0].([]interface{})
	beginTimestamp := int64(firstElemt[0].(float64))

	// 判断第一个元素的第二个元素的类型
	switch getElementType(firstElemt[1]) {
	// [1231231,[12,12]]
	case ELEMENT_TYPE_3_INT:
		// 将第一个数据添加进去
		finalData = append(finalData, []int64{int64(beginTimestamp), int64(firstElemt[1].([]interface{})[0].(float64)), int64(firstElemt[1].([]interface{})[1].(float64))})
		// [12,12] 或者 "NA"
		for i, o := range cData {
			if i == 0 {
				continue
			}
			sf := reflect.TypeOf(o).Kind()
			switch sf {
			case reflect.Array, reflect.Slice:
				currentElement := o.([]interface{})
				finalData = append(finalData, []int64{int64(beginTimestamp) + int64(i), int64(currentElement[0].(float64)), int64(currentElement[1].(float64))})
			default:
				// log.Printf("t %d type is %v", i, sf)
				continue
			}
		}
	case ELEMENT_TYPE_3_STRING:
		end := len(firstElemt)
		var x = make([]int64, 0)
		x = append(x, beginTimestamp)
		for y := 1; y < end; y++ {
			s1 := firstElemt[y].(string)
			s1Array := strings.Split(s1, "|")
			for _, o := range s1Array {
				d, _ := strconv.ParseInt(o, 10, 0)
				x = append(x, d)
			}
		}
		finalData = append(finalData, x)
		for i, o := range cData {
			if i == 0 {
				continue
			}
			sf := reflect.TypeOf(o).Kind()
			switch sf {
			case reflect.Array, reflect.Slice:
				currentElement := o.([]interface{})
				var x = make([]int64, 0)
				x = append(x, beginTimestamp+int64(i))
				for y := 0; y < end-1; y++ {
					s1 := currentElement[y].(string)
					s1Array := strings.Split(s1, "|")
					for _, o := range s1Array {
						d, _ := strconv.ParseInt(o, 10, 0)
						x = append(x, d)
					}
				}
				finalData = append(finalData, x)
			}
		}
	case ELEMENT_TYPE_2_INT:
		finalData = append(finalData, []int64{beginTimestamp, int64(firstElemt[1].(float64))})
		for i, o := range cData {
			if i == 0 {
				continue
			}
			sf := reflect.TypeOf(o).Kind()

			switch sf {
			case reflect.Float32, reflect.Float64, reflect.Int64:
				currentElement := int64(o.(float64))
				finalData = append(finalData, []int64{beginTimestamp + int64(i), currentElement})
			default:
				continue
			}
		}
	}
	return finalData
}

// [[1392072000, 1], 1, 2, 3, ... ]
// [[1392072000, [1,2]], "NA", [1,2], [500, value] , ... ]
// [[1381816200,"3345|90|0|0|2925|330|0|0|0|0|0", "1204|70|0|0|2232|330|0|0|0|0|0"]]
// TODO
const (
	ELEMENT_TYPE_3_INT    = 1
	ELEMENT_TYPE_3_STRING = 2
	ELEMENT_TYPE_2_INT    = 3
	ELEMENT_TYPE_2_STRING = 3
)

func getElementType(f interface{}) int {
	sf := reflect.TypeOf(f).Kind()
	switch sf {
	case reflect.String:
		return ELEMENT_TYPE_3_STRING
	case reflect.Slice:
		return ELEMENT_TYPE_3_INT
	default:
		return ELEMENT_TYPE_2_INT
	}
}
