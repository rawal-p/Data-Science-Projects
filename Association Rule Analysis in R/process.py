#heart_failure_clinical_records_dataset.csv content:

#1  age	                            int
#2  anaemia	                        bool
#3  creatinine_phosphokinase	    int
#4  diabetes	                    bool
#5  ejection_fraction	            int
#6  high_blood_pressure	            bool
#7  platelets	                    int
#8  serum_creatinine	            double
#9  serum_sodium	                int
#10 sex	                            bool
#11 smoking	                        bool
#12 time	                        int
#13 DEATH_EVENT                     bool 


#remove all non-boolean columns and save to heart_failure_bool.csv

fr = open('heart_fail_data.txt','w')
i = 1
j = 1
with open('heart_fail.txt','r') as f:
    for line in f:
        for word in line.split():
            if word == '1':
                word = 'y' + str(j-1)
            if word == '0':
                word = 'n' + str(j-1)
                
            fr.write(str(i)+ '\t' + word + '\n')
            j = j+1
            if j == 7:
                i = i+1
                j = 1
