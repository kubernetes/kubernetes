package gofuzzheaders

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"unsafe"

	securejoin "github.com/cyphar/filepath-securejoin"
)

type ConsumeFuzzer struct {
	data                 []byte
	CommandPart          []byte
	RestOfArray          []byte
	NumberOfCalls        int
	position             int
	fuzzUnexportedFields bool
}

func IsDivisibleBy(n int, divisibleby int) bool {
	return (n % divisibleby) == 0
}

func NewConsumer(fuzzData []byte) *ConsumeFuzzer {
	f := &ConsumeFuzzer{data: fuzzData, position: 0}
	return f
}

func (f *ConsumeFuzzer) Split(minCalls, maxCalls int) error {
	if len(f.data) == 0 {
		return errors.New("Could not split")
	}
	numberOfCalls := int(f.data[0])
	if numberOfCalls < minCalls || numberOfCalls > maxCalls {
		return errors.New("Bad number of calls")

	}
	if len(f.data) < numberOfCalls+numberOfCalls+1 {
		return errors.New("Length of data does not match required parameters")
	}

	// Define part 2 and 3 of the data array
	commandPart := f.data[1 : numberOfCalls+1]
	restOfArray := f.data[numberOfCalls+1:]

	// Just a small check. It is necessary
	if len(commandPart) != numberOfCalls {
		return errors.New("Length of commandPart does not match number of calls")
	}

	// Check if restOfArray is divisible by numberOfCalls
	if !IsDivisibleBy(len(restOfArray), numberOfCalls) {
		return errors.New("Length of commandPart does not match number of calls")
	}
	f.CommandPart = commandPart
	f.RestOfArray = restOfArray
	f.NumberOfCalls = numberOfCalls
	return nil
}

func (f *ConsumeFuzzer) AllowUnexportedFields() {
	f.fuzzUnexportedFields = true
}

func (f *ConsumeFuzzer) DisallowUnexportedFields() {
	f.fuzzUnexportedFields = false
}

func (f *ConsumeFuzzer) GenerateStruct(targetStruct interface{}) error {
	v := reflect.ValueOf(targetStruct)
	/*if !v.CanSet() {
		return errors.New("This interface cannot be set")
	}*/
	e := v.Elem()
	err := f.fuzzStruct(e)
	if err != nil {
		return err
	}
	return nil
}

func (f *ConsumeFuzzer) fuzzStruct(e reflect.Value) error {
	switch e.Kind() {
	case reflect.Struct:
		for i := 0; i < e.NumField(); i++ {
			var v reflect.Value
			if !e.Field(i).CanSet() {
				if f.fuzzUnexportedFields {
					v = reflect.NewAt(e.Field(i).Type(), unsafe.Pointer(e.Field(i).UnsafeAddr())).Elem()
				}
			} else {
				v = e.Field(i)
			}
			err := f.fuzzStruct(v)
			if err != nil {
				return err
			}
		}
	case reflect.String:
		str, err := f.GetString()
		if err != nil {
			return err
		}
		if e.CanSet() {
			e.SetString(str)
		}
	case reflect.Slice:
		maxElements := 50
		randQty, err := f.GetInt()
		if err != nil {
			return err
		}
		numOfElements := randQty % maxElements

		uu := reflect.MakeSlice(e.Type(), numOfElements, numOfElements)

		for i := 0; i < numOfElements; i++ {
			err := f.fuzzStruct(uu.Index(i))
			if err != nil {
				return err
			}
		}
		if e.CanSet() {
			e.Set(uu)
		}
	case reflect.Uint64:
		newInt, err := f.GetInt()
		if err != nil {
			return err
		}
		if e.CanSet() {
			e.SetUint(uint64(newInt))
		}
	case reflect.Int:
		newInt, err := f.GetInt()
		if err != nil {
			return err
		}
		if e.CanSet() {
			e.SetInt(int64(newInt))
		}
	case reflect.Map:
		if e.CanSet() {
			e.Set(reflect.MakeMap(e.Type()))
			maxElements := 50
			randQty, err := f.GetInt()
			if err != nil {
				return err
			}
			numOfElements := randQty % maxElements
			for i := 0; i < numOfElements; i++ {
				key := reflect.New(e.Type().Key()).Elem()
				err := f.fuzzStruct(key)
				if err != nil {
					return err
				}
				val := reflect.New(e.Type().Elem()).Elem()
				err = f.fuzzStruct(val)
				if err != nil {
					return err
				}
				e.SetMapIndex(key, val)
			}
		}
	case reflect.Ptr:
		if e.CanSet() {
			e.Set(reflect.New(e.Type().Elem()))
			err := f.fuzzStruct(e.Elem())
			if err != nil {
				return err
			}
			return nil
		}
	case reflect.Uint8:
		b, err := f.GetByte()
		if err != nil {
			return err
		}
		if e.CanSet() {
			e.SetUint(uint64(b))
		}
	default:
		return nil
	}
	return nil
}

func (f *ConsumeFuzzer) GetStringArray() (reflect.Value, error) {
	// The max size of the array:
	max := 20

	arraySize := f.position
	if arraySize > max {
		arraySize = max
	}
	elemType := reflect.TypeOf("string")
	stringArray := reflect.MakeSlice(reflect.SliceOf(elemType), arraySize, arraySize)
	if f.position+arraySize >= len(f.data) {
		return stringArray, errors.New("Could not make string array")
	}

	for i := 0; i < arraySize; i++ {
		stringSize := int(f.data[f.position])

		if f.position+stringSize >= len(f.data) {
			return stringArray, nil
		}
		stringToAppend := string(f.data[f.position : f.position+stringSize])
		strVal := reflect.ValueOf(stringToAppend)
		stringArray = reflect.Append(stringArray, strVal)
		f.position = f.position + stringSize
	}
	return stringArray, nil
}

func (f *ConsumeFuzzer) GetInt() (int, error) {
	if f.position >= len(f.data) {
		return 0, errors.New("Not enough bytes to create int")
	}
	returnInt := int(f.data[f.position])
	f.position++
	return returnInt, nil
}

func (f *ConsumeFuzzer) GetByte() (byte, error) {
	if len(f.data) == 0 {
		return 0x00, errors.New("Not enough bytes to get byte")
	}
	if f.position >= len(f.data) {
		return 0x00, errors.New("Not enough bytes to get byte")
	}
	returnByte := f.data[f.position]
	f.position += 1
	return returnByte, nil
}

func (f *ConsumeFuzzer) GetBytes() ([]byte, error) {
	if len(f.data) == 0 || f.position >= len(f.data) {
		return nil, errors.New("Not enough bytes to create byte array")
	}
	length := int(f.data[f.position])
	byteBegin := f.position + 1
	if byteBegin >= len(f.data) {
		return nil, errors.New("Not enough bytes to create byte array")
	}
	if length == 0 {
		return nil, errors.New("Zero-length is not supported")
	}
	if byteBegin+length >= len(f.data) {
		return nil, errors.New("Not enough bytes to create byte array")
	}
	b := f.data[byteBegin : byteBegin+length]
	f.position = byteBegin + length
	return b, nil
}

func (f *ConsumeFuzzer) GetString() (string, error) {
	if f.position >= len(f.data) {
		return "nil", errors.New("Not enough bytes to create string")
	}
	length := int(f.data[f.position])
	byteBegin := f.position + 1
	if byteBegin >= len(f.data) {
		return "nil", errors.New("Not enough bytes to create string")
	}
	if byteBegin+length > len(f.data) {
		return "nil", errors.New("Not enough bytes to create string")
	}
	str := string(f.data[byteBegin : byteBegin+length])
	f.position = byteBegin + length
	return str, nil
}

func (f *ConsumeFuzzer) GetBool() (bool, error) {
	if f.position >= len(f.data) {
		return false, errors.New("Not enough bytes to create bool")
	}
	if IsDivisibleBy(int(f.data[f.position]), 2) {
		f.position++
		return true, nil
	} else {
		f.position++
		return false, nil
	}
}

func (f *ConsumeFuzzer) FuzzMap(m interface{}) error {
	err := f.GenerateStruct(m)
	if err != nil {
		return err
	}
	return nil
}

// TarBytes returns valid bytes for a tar archive
func (f *ConsumeFuzzer) TarBytes() ([]byte, error) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)

	numberOfFiles, err := f.GetInt()
	if err != nil {
		return nil, err
	}
	maxNoOfFiles := 100000
	for i := 0; i < numberOfFiles%maxNoOfFiles; i++ {
		filename, err := f.GetString()
		if err != nil {
			return nil, err
		}
		filebody, err := f.GetBytes()
		if err != nil {
			return nil, err
		}
		hdr := &tar.Header{}
		err = f.GenerateStruct(hdr)
		if err != nil {
			return nil, err
		}
		hdr.Name = filename
		hdr.Size = int64(len(filebody))
		hdr.Mode = 0600

		if err := tw.WriteHeader(hdr); err != nil {
			return nil, err
		}
		if _, err := tw.Write(filebody); err != nil {
			return nil, err
		}
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Creates pseudo-random files in rootDir.
// Will create subdirs and place the files there.
// It is the callers responsibility to ensure that
// rootDir exists.
func (f *ConsumeFuzzer) CreateFiles(rootDir string) error {

	noOfCreatedFiles := 0
	numberOfFiles, err := f.GetInt()
	if err != nil {
		return err
	}
	maxNumberOfFiles := numberOfFiles % 4000 // This is completely arbitrary
	if maxNumberOfFiles == 0 {
		return errors.New("maxNumberOfFiles is nil")
	}

	for i := 0; i < maxNumberOfFiles; i++ {
		subDir, err := f.GetString()
		if err != nil {
			// We don't stop the fuzzing job if we have created
			// some files. In that case we want to feed them
			// to the target
			if noOfCreatedFiles > 0 {
				return nil
			}
		}

		// Avoid going outside the root dir
		if strings.Contains(subDir, "../") || (len(subDir) > 0 && subDir[0] == 47) || strings.Contains(subDir, "\\") {
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Dir contains invalid chars")
			}
		}

		dirPath, err := securejoin.SecureJoin(rootDir, subDir)
		if err != nil {
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not join rootDir and subDir securely")
			}
		}
		err = os.MkdirAll(dirPath, 0777)
		if err != nil {
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not create the subDir")
			}
		}
		fileName, err := f.GetString()
		if err != nil {
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not get fileName")
			}
		}
		fullFilePath, err := securejoin.SecureJoin(dirPath, fileName)
		if err != nil {
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not join dir path and file path securely")
			}
		}
		fileContents, err := f.GetBytes()
		if err != nil {
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not create the subDir")
			}
		}

		createdFile, err := os.Create(fullFilePath)
		if err != nil {
			createdFile.Close()
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not create the subDir")
			}
		}
		_, err = createdFile.Write(fileContents)
		if err != nil {
			createdFile.Close()
			if noOfCreatedFiles > 0 {
				return nil
			} else {
				return errors.New("Could not create the subDir")
			}
		}
		createdFile.Close()
		noOfCreatedFiles++
	}
	return nil
}

// Returns a string that can only consists of characters that are
// included in possibleChars. Will return an error if the created
// string does not have the specified length
func (f *ConsumeFuzzer) GetStringFrom(possibleChars string, length int) (string, error) {
	returnString := ""
	if (len(f.data)-f.position)<length {
		return returnString, errors.New("Not enough bytes to create a string")
	}
	for i:=0;i<length;i++ {
		charIndex, err := f.GetInt()
		if err != nil {
			return returnString, err
		}
		charToAdd := string(possibleChars[charIndex%len(possibleChars)])
		returnString = fmt.Sprintf(returnString+charToAdd)
	}
	return returnString, nil
}