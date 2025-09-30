// Pin untuk sensor ultrasonik kiri
const int trigKiri = 2;
const int echoKiri = 3;

// Pin untuk sensor ultrasonik tengah
const int trigTengah = 4;
const int echoTengah = 5;

// Pin untuk sensor ultrasonik kanan
const int trigKanan = 6;
const int echoKanan = 7;

void setup() {
  Serial.begin(115200);

  pinMode(trigKiri, OUTPUT);
  pinMode(echoKiri, INPUT);

  pinMode(trigTengah, OUTPUT);
  pinMode(echoTengah, INPUT);

  pinMode(trigKanan, OUTPUT);
  pinMode(echoKanan, INPUT);
}

float bacaJarak(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long durasi = pulseIn(echoPin, HIGH, 25000); // timeout 30 ms

  if (durasi == 0) {
    return 999.99; // tidak ada respons, dianggap objek terlalu jauh
  }

  float jarak = durasi * 0.0343 / 2.0;

  if (jarak > 250.0) {
    return 999.99; // objek terlalu jauh
  }

  return jarak;
}

void loop() {
  float jarakKiri = bacaJarak(trigKiri, echoKiri);
  float jarakTengah = bacaJarak(trigTengah, echoTengah);
  float jarakKanan = bacaJarak(trigKanan, echoKanan);

  Serial.print(jarakKiri, 2);
  Serial.print(",");
  Serial.print(jarakTengah, 2);
  Serial.print(",");
  Serial.println(jarakKanan, 2);

  delay(250);
}