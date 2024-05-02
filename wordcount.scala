val input = sc.textFile("passage.txt")
val words = input.flatMap(x => x.split(" "))
val counts = words.map(x => (x, 1))
val reducedCounts = counts.reduceByKey((x, y) => x + y)
reducedCounts.saveAsTextFile("output.txt")
reducedCounts.foreach(println)

go to spark folder
go in bin
put wordcount.scala and input text file in bin folder
open in terminal

./spark-shell
:load wordcount.scala