echo "Starting training processes..."
for i in {0..9}; do
    python3 fine.py --index $i &
    echo "Started training process for index $i"
done
wait
echo "All training processes have completed."