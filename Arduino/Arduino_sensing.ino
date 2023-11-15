   

   
   //****************************************************************************************
// Illutron take on Disney style capacitive touch sensor using only passives and Arduino
// Dzl 2012
//****************************************************************************************


//                              10n
// PIN 9 --[10k]-+-----10mH---+--||-- OBJECT
//               |            |
//              3.3k          |
//               |            V 1N4148 diode
//              GND           |
//                            |
//Analog 0 ---+------+--------+
//            |      |
//          100pf   1MOmhm
//            |      |
//           GND    GND



#define SET(x,y) (x |=(1<<y))				//-Bit set/clear macros
#define CLR(x,y) (x &= (~(1<<y)))       		// |
#define CHK(x,y) (x & (1<<y))           		// |
#define TOG(x,y) (x^=(1<<y))            		//-+



#define N 160  //How many frequencies
unsigned int measureTime = 1000;

float results[N];            //-Filtered result buffer
float freq[N];            //-Filtered result buffer
int sizeOfArray = N;
char input;
unsigned int d; //Initialize loop counter
unsigned int it_counter; //Counter for while loop later
unsigned long startTime;
int counter;
int i;
int v;
   
   

void setup()
{
  
  
  TCCR1A=0b10000010;        //-Set up frequency generator
  TCCR1B=0b00011001;        //-+
  ICR1=110;
  OCR1A=55;

  pinMode(9,OUTPUT);        //-Signal generator pin
  pinMode(8,OUTPUT);        //-Sync (test) pin

  Serial.begin(115200);

  reset_results();
}

void reset_results()
{
  for(i=0;i<N;i++)      //-Preset results
    results[i]=0;         //-+
}

void loop()
{
  //measure();
  //reset_results();
  
  if(Serial.available()){  // Dit is mijn stukje code. Ja ik ben een nester nester. Ik pak je in Ping Pong als je niet eens bent!
    input = Serial.read();
    if (input == 88 ) {  // Als X wordt ontvangen, dan leest hij de detector uit!
      startTime = millis();
      it_counter = 0;
      while( millis() - startTime < measureTime){
        measure();
        it_counter++;
      }
      //Now divide results by it_counter to finally take the mean
      for(d=0; d<N; d++)
      {
        results[d] /= it_counter;
      }
      //Having taken the mean, we can now send the results through serial
      PlottArray(freq,results);
    }
  }
}

void measure()
{
  counter = 0;
    for(d=0;d<N;d++)
    {
      v=analogRead(0);    //-Read response signal
      CLR(TCCR1B,0);          //-Stop generator
      TCNT1=0;                //-Reload new frequency
      ICR1=d;                 // |
      OCR1A=d/2;              //-+
      SET(TCCR1B,0);          //-Restart generator

      results[d] += (float)(v); //Filter results
    
      freq[d] = d;
    }
    TOG(PORTB,0);            //-Toggle pin 8 after each sweep (good for scope)
}
