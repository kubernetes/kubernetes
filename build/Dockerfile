FROM golang:1.20-alpine

WORKDIR /app

COPY . .

# Download dependencies
RUN go mod tidy

# Build the Go app
RUN go build -o dockergin .

# Expose port
EXPOSE 8000

# Run the binary
ENTRYPOINT ["./dockergin"]