def enumerate_esplain_via_print(bool):
    if bool == 1:
        my_list = [0,1,2,3] #example list
        for index, content in enumerate(my_list):#here calls with enumerate() instead of range(len())
            print(f"Index: {index}, Content: {content}")#and content is directly available, iterable[index] is not needed
        #alternative way to do the same thing
        for index in range(len(my_list)):
            print(f"Index: {index}, Content: {my_list[index]}")
        for index, content in enumerate(my_list, start=1):#enumerate can also take a start argument
            print(f"Index: {index}, Content: {content}")#Index: 0, Content: 0 will be skipped
enumerate_esplain_via_print(True)



list1 = [1, 2, 3]  
print(sum(list1))  # Output: 6