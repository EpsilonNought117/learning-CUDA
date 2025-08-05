#include <stdio.h>
#include <stdlib.h>

typedef struct sll_node
{
	int data;
	struct sll_node* next;

}	sll_node;

void addNodeAtEnd(sll_node** head, int val)
{
	sll_node* myNode = (sll_node*)malloc(sizeof(sll_node));

	if (!myNode)
	{
		fprintf(stderr, "Memory alloc failure in addNodeAtEnd!");
		abort();
	}

	myNode->data = val;
	myNode->next = NULL;

	if (!(*head))
	{
		(*head) = myNode;
		return;
	}

	sll_node* curr_ptr = (*head);

	while (curr_ptr && curr_ptr->next != NULL)
	{
		curr_ptr = curr_ptr->next;
	}

	curr_ptr->next = myNode;
	return;
}

void deleteNodeAtEnd(sll_node** head)
{
	if (!(*head))
	{
		fprintf(stderr, "Nothing to delete in deleteNodeAtEnd!");
		abort();
	}

	sll_node* prev_node;
	sll_node* curr_node;

	if ((*head)->next == NULL)
	{
		curr_node = (*head);
		*head = NULL;
		free(curr_node);
		return;
	}

	prev_node = (*head);
	curr_node = (*head)->next;

	while (curr_node->next != NULL)
	{
		prev_node = prev_node->next;
		curr_node = curr_node->next;
	}

	prev_node->next = curr_node->next;
	free(curr_node);
	return;
}

void printList(sll_node* head)
{
	while (head)
	{
		printf("%d -> ", head->data);
		head = head->next;
	}
	printf("NULL\n");
}