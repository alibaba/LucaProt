dataset_name=""
dataset_type=""
task_type=""
model_type=""
time_str=""

if [ $# -le 4 ]
then
  echo 'params num < 5'
  exit
fi

if [ $# -ge 1 ]
then
  dataset_name="/"$1
fi
if [ $# -ge 2 ]
then
  dataset_type="/"$2
fi
if [ $# -ge 3 ]
then
  task_type="/"$3
fi
if [ $# -ge 4 ]
then
  model_type="/"$4
fi
if [ $# -ge 5 ]
then
  time_str="/"$5
fi
echo ../../logs$dataset_name$dataset_type$task_type$model_type$time_str
echo ../../tb-logs$dataset_name$dataset_type$task_type$model_type$time_str
echo ../../models$dataset_name$dataset_type$task_type$model_type$time_str

rm -rf ../../logs$dataset_name$dataset_type$task_type$model_type$time_str
rm -rf ../../tb-logs$dataset_name$dataset_type$task_type$model_type$time_str
rm -rf ../../models$dataset_name$dataset_type$task_type$model_type$time_str
