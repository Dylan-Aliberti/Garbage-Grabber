    byte yLSB=0, yMSB=0, xLSB=0, zeroByte=128;  
 
    void SendData(unsigned int xValue,unsigned int yValue){
    

      
      /* >=================================================================< 
          y = 01010100 11010100    (x & y are 2 Byte integers)
               yMSB      yLSB      send seperately -> reciever joins them
        >=================================================================<  */
       
        xLSB=lowByte(xValue);
        yLSB=lowByte(yValue);
        yMSB=highByte(yValue);        
   
      
     /* >=================================================================< 
        Only the very first Byte may be a zero, this way allows the computer 
        to know that if a Byte recieved is a zero it must be the start byte.
        If data bytes actually have a value of zero, They are given the value 
        one and the bit in the zeroByte that represents that Byte is made 
        high.  
        >=================================================================< */   
        
       zeroByte = 128;                                   // 10000000
   
       if(xLSB==0){ xLSB=1; zeroByte=zeroByte+1;}        // Make bit 1 high 
       //if(yMSB==0){ yMSB=1; zeroByte=zeroByte+2;}        // make bit 2 high
       if(yLSB==0){ yLSB=1; zeroByte=zeroByte+4;}        // make bit 3 high
       if(yMSB==0){ yMSB=1; zeroByte=zeroByte+8;}        // make bit 4 high

        Serial.write(byte(xLSB));         // Y value's least significant byte   
        Serial.write(byte(yMSB));         // X value's most significant byte  
        Serial.write(byte(yLSB));         // X value's least significant byte  
        Serial.write(byte(zeroByte));     // Which values have a zero value
    }  
    


       
 void PlottArray(float Array1[],float Array2[]){
   
      SendData(1,1);                        // Tell PC an array is about to be sent                      
      delay(1);
      for(int x=0;  x < sizeOfArray;  x++){     // Send the arrays 
        SendData(round(Array1[x]),round(Array2[x]));
        //delay(1);
      }
      
      SendData(1,1);                        // Confirm arrrays have been sent
    }
