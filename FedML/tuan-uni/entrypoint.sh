python download.py
python initialize.py 
source data_config

if [[ $framework = 'horizontal' ]]
then
    python main_fedavg.py
elif [[ $framework = 'vertical' ]]
then 
    python vertical_exp.py

else 
    echo 'dataset error!'
fi
